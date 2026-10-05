"""A loopback-only, persistent, single-worker web interface for trFlow."""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import io
import json
import mimetypes
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .config import REPO, MODEL_NAMES, RunOptions
from .web_errors import WebRunError, error_fields

STATIC = Path(__file__).parent / "web_static"
ACTIVE = {"queued", "running", "analyzing"}
MAX_BODY = 64 * 1024 * 1024
STAGES = {0: "等待运行", 1: "初始结构与表征", 2: "几何探索", 3: "构象生成", 4: "结构对齐与聚类", 5: "运行完成"}


class LocalHTTPServer(ThreadingHTTPServer):
    """Do not let a second Windows listener reuse an occupied workbench port."""

    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def tail(path: Path, lines: int = 100) -> str:
    if not path.is_file():
        return ""
    # Only read the tail even for long-running exploration logs.
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 32768))
        text = handle.read().decode("utf-8", errors="replace")
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(text.splitlines()[-lines:]))


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("路径必须位于当前任务目录内")
    return path


def example_target() -> dict:
    msa = REPO / "example" / "msa" / "2akl.a3m"
    if not msa.is_file():
        # A wheel carries the same example without depending on a source checkout.
        msa = Path(__file__).parent / "web_assets" / "example" / "2akl.a3m"
    return {"name": msa.stem, "msa_filename": msa.name, "msa_text": msa.read_text(encoding="utf-8")}


def default_data_dir() -> Path:
    """Keep checkout data local; never write into installed site-packages."""
    source_checkout = (REPO / "pyproject.toml").is_file() and (REPO / "trflow" / "web.py").is_file()
    root = REPO if source_checkout else Path.cwd()
    return root / "outputs" / "web"


def download_filename(record: dict) -> str:
    """Use the user-facing target name, never the internal queue UUID."""
    name = str(record.get("name") or record.get("sample_name") or "target").strip()
    name = re.sub(r"[^\w-]+", "_", name).strip("_")[:120] or "target"
    return f"trflow_{name}.zip"


class TargetConflictError(ValueError):
    """A recoverable target state conflict, exposed as HTTP 409."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


class TargetStore:
    """SQLite-backed records; every mutation uses a separate transaction."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS targets (id TEXT PRIMARY KEY, created TEXT NOT NULL, data TEXT NOT NULL)")
        self.lock = threading.RLock()

    def list(self, include_deleted: bool = False) -> list[dict]:
        with self.lock, closing(sqlite3.connect(self.path)) as db:
            records = [json.loads(row[0]) for row in db.execute("SELECT data FROM targets ORDER BY created, rowid")]
        return records if include_deleted else [record for record in records if not record.get("deleted_at")]

    def get(self, target_id: str, include_deleted: bool = False) -> dict:
        with self.lock, closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT data FROM targets WHERE id=?", (target_id,)).fetchone()
        if row is None:
            raise KeyError("目标不存在")
        record = json.loads(row[0])
        if record.get("deleted_at") and not include_deleted:
            raise KeyError("目标不存在")
        return record

    def add_many(self, records: list[dict]) -> None:
        with self.lock, closing(sqlite3.connect(self.path)) as db, db:
            db.executemany("INSERT INTO targets(id,created,data) VALUES (?,?,?)", [
                (record["id"], record["created_at"], json.dumps(record, ensure_ascii=False)) for record in records
            ])

    def update(self, target_id: str, **changes) -> dict:
        with self.lock:
            record = self.get(target_id, include_deleted=True)
            record.update(changes)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("UPDATE targets SET data=? WHERE id=?", (json.dumps(record, ensure_ascii=False), target_id))
            return record


class JobManager:
    def __init__(self, data_dir: Path, interpreter: str = sys.executable, env: str | None = None,
                 import_existing: bool = True, start_worker: bool = True):
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.service_lock = (self.data_dir / "service.lock").open("a+b")
        try:
            self.service_lock.seek(0, 2)
            if self.service_lock.tell() == 0:
                self.service_lock.write(b"1")
                self.service_lock.flush()
            self.service_lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.service_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.service_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.service_lock.close()
            raise RuntimeError("此数据目录已有 trFlow 网页服务运行，请使用现有服务。") from exc
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        self.recycle_dir = self.data_dir / "recycle_bin"
        self.interpreter = interpreter
        self.env = str(Path(env).resolve()) if env else None
        self.store = TargetStore(self.data_dir / "targets.sqlite")
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.process_lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.running_id: str | None = None
        self.analysis_locks: dict[str, threading.RLock] = {}
        self.analysis_processes: dict[str, subprocess.Popen] = {}
        self.analysis_lock = threading.Lock()
        self.submit_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        for record in self.store.list():
            if record["status"] in {"running", "analyzing"}:
                if self._prediction_ready(record):
                    self.store.update(record["id"], status="queued", stage=4, stage_label="等待继续结构分析",
                                      error=None, error_code=None, error_params={}, finished_at=None)
                else:
                    self.store.update(record["id"], status="failed", finished_at=now(),
                                      **error_fields("服务已中断；可重新提交此目标。"))
        if import_existing:
            self._import_existing()
        if start_worker:
            self.thread = threading.Thread(target=self._worker, name="trflow-single-worker", daemon=True)
            self.thread.start()

    def _import_existing(self) -> None:
        # Existing curated results are read-only. Analysis files go into web data.
        for info_file in sorted((REPO / "outputs").glob("*/info.json")):
            source = info_file.parent
            if source.is_relative_to(self.data_dir):
                continue
            target_id = "existing-" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
            try:
                existing = self.store.get(target_id, include_deleted=True)
            except KeyError:
                existing = None
            try:
                info = read_json(info_file)
                predictions = sorted((source / "predictions").glob("*.pdb"))
                if not predictions:
                    continue
                length = info.get("sequence", {}).get("length")
                if not length:
                    residues = set()
                    for line in predictions[0].read_text(encoding="utf-8").splitlines():
                        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                            residues.add((line[21:22], line[22:26], line[26:27]))
                    length = len(residues)
                timing = info.get("stage_times", {}).get("total", {})
                metadata = {
                    "length": length, "started_at": timing.get("start"),
                    "finished_at": timing.get("end"), "elapsed_seconds": timing.get("seconds"),
                }
                if existing:
                    if not existing.get("deleted_at"):
                        self.store.update(target_id, **metadata)
                    continue
                record = {
                    "id": target_id, "name": info.get("sample_name", source.name), "sample_name": source.name,
                    "status": "completed", "stage": 5, "stage_label": STAGES[5], "generated": len(predictions),
                    "sample_num": len(predictions), "length": length,
                    "created_at": datetime.fromtimestamp(info_file.stat().st_mtime, timezone.utc).isoformat(),
                    "started_at": None, "finished_at": None, "error": None, "existing": True,
                    "options": info.get("options", {}), "output_path": str(source),
                    "job_path": str(self.jobs_dir / target_id), "queue_position": None,
                }
                record.update(metadata)
                self.store.add_many([record])
            except (ValueError, OSError, sqlite3.Error):
                continue

    def config(self) -> dict:
        from .config import load_env_config
        try:
            env = load_env_config(self.env)
            paths = [env.trflow_xray, env.trflow_nmr, env.esm_weights, env.openfold.param_path]
            ready = all(Path(path).is_file() for path in paths)
        except (ValueError, OSError):
            ready = False
        return {"defaults": dataclasses.asdict(RunOptions()), "models_ready": ready,
                "device": "local", "single_worker": True, "data_dir": str(self.data_dir),
                "max_targets": 50, "max_conformations": 2000, "version": "0.1.0"}

    @staticmethod
    def _options(raw: dict | None) -> dict:
        raw = {} if raw is None else raw
        if not isinstance(raw, dict):
            raise ValueError("options 必须是 JSON 对象")
        known = {field.name for field in dataclasses.fields(RunOptions)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError("未知参数：" + ", ".join(sorted(unknown)))
        options = dataclasses.asdict(RunOptions())
        options.update(raw)
        models = options["models"]
        if isinstance(models, str):
            models = [item.strip() for item in models.split(",") if item.strip()]
        if not isinstance(models, list) or not models or any(model not in MODEL_NAMES for model in models):
            raise ValueError("models 请选择 Xray、NMR 或两者")
        options["models"] = list(dict.fromkeys(models))
        for key in ("sample_num", "steps"):
            value = options[key]
            if type(value) is not int or not 1 <= value <= (2000 if key == "sample_num" else 100):
                raise ValueError(f"{key} 必须为有效正整数")
        for key in ("geometric_exploration", "single_step", "random_step", "random_step_size", "parallel", "save_repr_npz"):
            if type(options[key]) is not bool:
                raise ValueError(f"{key} 必须为 true 或 false")
        if options["seed"] is not None and (type(options["seed"]) is not int or not 0 <= options["seed"] < 2**32):
            raise ValueError("seed 为空或为 0 至 4294967295 之间的整数")
        if options["gpus"] is not None:
            if not isinstance(options["gpus"], list) or not options["gpus"] or any(type(x) is not int or x < 0 for x in options["gpus"]):
                raise ValueError("gpus 必须为 GPU 编号数组")
        if options["max_workers"] is not None and (type(options["max_workers"]) is not int or options["max_workers"] < 1):
            raise ValueError("max_workers 必须为正整数或空")
        return options

    def _prepare(self, spec: dict) -> dict:
        if not isinstance(spec, dict):
            raise ValueError("每个 target 必须是 JSON 对象")
        name = str(spec.get("name", "")).strip()
        if not name or len(name) > 120:
            raise ValueError("目标名称必须为 1 至 120 个字符")
        sample_name = re.sub(r"[^\w-]+", "_", name).strip("_") or "target"
        # Windows reserves device names even when a filename has an extension.
        # Keep the display name, but use a portable name for generated files.
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]", sample_name, re.IGNORECASE):
            sample_name = f"target_{sample_name}"
        msa = spec.get("msa_text")
        if not isinstance(msa, str) or not msa.strip():
            raise ValueError(f"{name} 缺少 A3M 内容")
        query = []
        header = None
        for line in msa.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{name} 的 A3M 首条记录标题为空")
            elif header is None:
                raise ValueError(f"{name} 的 A3M 必须以 > 标题开头")
            else:
                query.append(line)
        sequence = "".join(query)
        if not sequence or any(char not in "ACDEFGHIKLMNPQRSTVWYX" for char in sequence):
            raise ValueError(f"{name} 的 A3M 首条记录必须为无 gap、无小写插入的目标序列")
        fasta = spec.get("fasta_text") or ""
        if fasta:
            if not isinstance(fasta, str) or not fasta.lstrip().startswith(">"):
                raise ValueError(f"{name} 的 FASTA 格式不正确")
            fasta_sequence = "".join(line.strip() for line in fasta.splitlines()[1:] if not line.startswith(">")).upper()
            if fasta_sequence != sequence:
                raise ValueError(f"{name} 的 FASTA 与 A3M 首条序列不一致")
        init = spec.get("init_pdb_text") or ""
        if init and (not isinstance(init, str) or not any(line.startswith("ATOM") for line in init.splitlines())):
            raise ValueError(f"{name} 的初始 PDB 缺少 ATOM 记录")
        options = self._options(spec.get("options"))
        if spec.get("seed") is not None:
            options = self._options({**options, "seed": spec["seed"]})
        return {"name": name, "sample_name": sample_name, "msa_text": msa, "fasta_text": fasta,
                "init_pdb_text": init, "length": len(sequence), "options": options}

    def submit(self, specs: list[dict]) -> list[dict]:
        if not isinstance(specs, list) or not 1 <= len(specs) <= 50:
            raise ValueError("一次可提交 1 至 50 个 target")
        # Validate the whole batch before writing anything or enqueuing a target.
        prepared = [self._prepare(spec) for spec in specs]
        records = []
        with self.submit_lock:
            for item in prepared:
                target_id = uuid.uuid4().hex
                job_path = self.jobs_dir / target_id
                inputs = job_path / "inputs"
                inputs.mkdir(parents=True)
                sample = {"name": item["sample_name"], "msa_path": str(inputs / "alignment.a3m")}
                (inputs / "alignment.a3m").write_text(item["msa_text"], encoding="utf-8")
                if item["fasta_text"]:
                    (inputs / "sequence.fasta").write_text(item["fasta_text"], encoding="utf-8")
                    sample["fasta_path"] = str(inputs / "sequence.fasta")
                if item["init_pdb_text"]:
                    (inputs / "initial.pdb").write_text(item["init_pdb_text"], encoding="utf-8")
                    sample["init_pdb"] = str(inputs / "initial.pdb")
                output = job_path / "output"
                cfg = {"output_dir": str(output), "samples": [sample], "options": item["options"]}
                write_json(job_path / "request.json", cfg)
                record = {
                    "id": target_id, "name": item["name"], "sample_name": item["sample_name"],
            "status": "queued", "stage": 0, "stage_label": STAGES[0], "generated": 0,
                    "sample_num": item["options"]["sample_num"], "length": item["length"],
                    "created_at": now(), "started_at": None, "finished_at": None, "error": None,
                    "options": item["options"], "existing": False, "job_path": str(job_path),
                    "output_path": str(output / item["sample_name"]), "queue_position": None,
                }
                records.append(record)
            self.store.add_many(records)
        self.wake.set()
        return [self.public(record) for record in records]

    def import_config(self, payload: dict) -> list[dict]:
        config = payload.get("config")
        if isinstance(config, str):
            config = json.loads(config)
        if not isinstance(config, dict) or not isinstance(config.get("samples"), list):
            raise ValueError("JSON 需包含 samples 数组；可使用 trFlow 原有输入格式")
        files = payload.get("files", [])
        if not isinstance(files, list):
            raise ValueError("files 必须是数组")
        uploaded = {}
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("content"), str):
                raise ValueError("上传文件需包含 name 与 content")
            filename = item["name"].replace("\\", "/").split("/")[-1]
            if filename in uploaded:
                raise ValueError(f"上传文件名称重复：{filename}")
            uploaded[filename] = item["content"]

        def content(sample: dict, field: str, inline: str, required=False) -> str:
            if sample.get(inline):
                return sample[inline]
            value = sample.get(field)
            if not value:
                if required:
                    raise ValueError(f"{sample.get('name', 'target')} 缺少 {field}")
                return ""
            name = str(value).replace("\\", "/").split("/")[-1]
            if name in uploaded:
                return uploaded[name]
            path = (REPO / str(value)).resolve()
            if not path.is_relative_to(REPO) and not path.is_relative_to(self.data_dir):
                raise ValueError(f"本地路径须位于项目目录内；或将文件一并上传：{value}")
            if not path.is_file():
                raise ValueError(f"找不到 {value}；请在页面上传相应文件")
            return path.read_text(encoding="utf-8")

        specs = []
        for sample in config["samples"]:
            if not isinstance(sample, dict):
                raise ValueError("samples 中每个元素必须为对象")
            specs.append({
                "name": sample.get("name", ""), "seed": sample.get("seed"),
                "msa_text": content(sample, "msa_path", "msa_text", True),
                "fasta_text": content(sample, "fasta_path", "fasta_text"),
                "init_pdb_text": content(sample, "init_pdb", "init_pdb_text"),
                "options": config.get("options", {}),
            })
        # Web output is always confined to per-job folders; original output_dir is not used.
        return self.submit(specs)

    def public(self, record: dict, logs: bool = False) -> dict:
        result = {key: value for key, value in record.items()
                  if key not in {"job_path", "output_path", "recycle_path", "deleted_at"}}
        if record.get("error"):
            result.update(error_fields(record["error"], record.get("error_code"), record.get("error_params")))
        else:
            result.update(error_code=None, error_params={})
        queued = [item["id"] for item in self.store.list() if item["status"] == "queued"]
        result["queue_position"] = queued.index(record["id"]) + 1 if record["id"] in queued else None
        if record["status"] in {"running", "analyzing"}:
            result["generated"] = len(list((Path(record["output_path"]) / "predictions").glob("*.pdb")))
        if logs:
            result["log_tail"] = self.log(record)
        return result

    def log(self, record: dict, full: bool = False) -> str:
        def read(path: Path, lines: int = 100) -> str:
            if not full:
                return tail(path, lines)
            if not path.is_file():
                return ""
            return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))

        log = read(Path(record["job_path"]) / "run.log")
        openfold = read(Path(record["output_path"]) / "openfold_log.txt", 25)
        return log + ("\n\n── OpenFold ──\n" + openfold if openfold else "")

    def target_lock(self, target_id: str):
        with self.analysis_lock:
            return self.analysis_locks.setdefault(target_id, threading.RLock())

    def _job_path(self, record: dict) -> Path:
        """Only move the exact, owned per-target directory, never external results."""
        target_id = record["id"]
        if not re.fullmatch(r"(?:[0-9a-f]{32}|existing-[0-9a-f]{16})", target_id):
            raise ValueError("Invalid target identifier")
        job = Path(record["job_path"]).resolve()
        expected = self.jobs_dir.resolve() / target_id
        if job != expected or not job.is_relative_to(self.data_dir):
            raise ValueError("The target directory is outside the workbench")
        if not record["existing"] and not Path(record["output_path"]).resolve().is_relative_to(job):
            raise ValueError("The output directory is outside this target")
        return job

    def delete(self, target_id: str) -> dict:
        record = self.store.get(target_id)
        if record["status"] in ACTIVE:
            raise TargetConflictError("Cancel this target before deleting it", "target_active")
        lock = self.target_lock(target_id)
        if not lock.acquire(blocking=False):
            raise TargetConflictError("The target is still releasing resources; try again shortly", "target_busy")
        try:
            with self.process_lock:
                record = self.store.get(target_id)
                if record["status"] in ACTIVE:
                    raise TargetConflictError("Cancel this target before deleting it", "target_active")
                if self.running_id == target_id or target_id in self.analysis_processes:
                    raise TargetConflictError("The target is still releasing resources; try again shortly", "target_busy")
                job = self._job_path(record)
                entry = contained(self.recycle_dir, target_id)
                if not entry.is_relative_to(self.data_dir):
                    raise ValueError("The recycle bin is outside the workbench")
                if entry.exists():
                    raise TargetConflictError("A recycle-bin entry already exists for this target", "recycle_conflict")
                entry.mkdir(parents=True)
                moved = False
                deleted_at = now()
                manifest = entry / "manifest.json"
                try:
                    write_json(manifest, {"record": record, "deleted_at": deleted_at, "job_moved": job.is_dir()})
                    if job.is_dir():
                        job.rename(entry / "job")
                        moved = True
                    self.store.update(target_id, deleted_at=deleted_at, recycle_path=str(entry))
                except Exception:
                    if moved:
                        (entry / "job").rename(job)
                    manifest.unlink(missing_ok=True)
                    manifest.with_suffix(".json.tmp").unlink(missing_ok=True)
                    entry.rmdir()
                    raise
            return {"deleted": True, "id": target_id, "can_restore": True}
        finally:
            lock.release()

    def recycled(self) -> list[dict]:
        keys = ("id", "name", "deleted_at", "existing", "status")
        return [{key: record.get(key) for key in keys}
                for record in self.store.list(include_deleted=True) if record.get("deleted_at")]

    def restore(self, target_id: str) -> dict:
        if not self.store.get(target_id, include_deleted=True).get("deleted_at"):
            raise TargetConflictError("This target is not in the recycle bin", "not_deleted")
        with self.target_lock(target_id), self.process_lock:
            record = self.store.get(target_id, include_deleted=True)
            if not record.get("deleted_at"):
                raise TargetConflictError("This target is not in the recycle bin", "not_deleted")
            job = self._job_path(record)
            entry = contained(self.recycle_dir, target_id)
            if not entry.is_relative_to(self.data_dir):
                raise ValueError("The recycle bin is outside the workbench")
            manifest = read_json(entry / "manifest.json")
            if manifest.get("record", {}).get("id") != target_id:
                raise ValueError("The recycle-bin metadata does not match this target")
            if job.exists():
                raise TargetConflictError("Restoring would overwrite an existing target directory", "restore_conflict")
            recycled_job = contained(entry, "job")
            if manifest["job_moved"] and not recycled_job.is_dir():
                raise FileNotFoundError("The recycled target directory is missing")
            moved = False
            audit_entry = contained(self.recycle_dir, f"{target_id}.restored-{uuid.uuid4().hex[:8]}")
            entry_moved = False
            try:
                if manifest["job_moved"]:
                    job.parent.mkdir(parents=True, exist_ok=True)
                    recycled_job.rename(job)
                    moved = True
                # Preserve our metadata without blocking future delete/restore cycles.
                # Do this before the database commit so any failure can roll back.
                entry.rename(audit_entry)
                entry_moved = True
                restored = self.store.update(target_id, deleted_at=None, recycle_path=None, restored_at=now())
            except Exception:
                if entry_moved:
                    audit_entry.rename(entry)
                if moved:
                    job.rename(recycled_job)
                raise
            return self.public(restored)

    def cancel(self, target_id: str) -> dict:
        with self.process_lock:
            record = self.store.get(target_id)
            if record["status"] not in ACTIVE:
                raise ValueError("只有等待或运行中的目标可以取消")
            self.store.update(target_id, status="cancelled", error=None, error_code=None, error_params={}, finished_at=now())
            if self.running_id == target_id and self.process is not None:
                self._terminate(self.process)
            if target_id in self.analysis_processes:
                self._terminate(self.analysis_processes[target_id])
        return self.public(self.store.get(target_id))

    def retry(self, target_id: str) -> list[dict]:
        if self.store.get(target_id)["status"] in ACTIVE:
            raise ValueError("目标仍在等待或运行中")
        with self.target_lock(target_id):
            return self._retry(target_id)

    def _retry(self, target_id: str) -> list[dict]:
        record = self.store.get(target_id)
        if record["existing"]:
            raise ValueError("已有结果无法重试，请从 MSA 重新提交")
        if record["status"] in ACTIVE:
            raise ValueError("目标仍在等待或运行中")
        if self._prediction_ready(record):
            result = self.store.update(target_id, status="queued", stage=4, stage_label="等待继续结构分析",
                                       error=None, error_code=None, error_params={}, finished_at=None)
            self.wake.set()
            return [self.public(result)]
        cfg = read_json(Path(record["job_path"]) / "request.json")
        return self.import_config({"config": cfg})

    @staticmethod
    def _prediction_ready(record: dict) -> bool:
        try:
            source = Path(record["output_path"])
            info = read_json(source / "info.json")
            entries = info.get("predictions", [])
            return len(entries) == record["sample_num"] and all(
                contained(source / "predictions", entry["file"]).is_file() for entry in entries
            )
        except (ValueError, KeyError, OSError):
            return False

    def _finish_analysis(self, record: dict) -> None:
        target_id = record["id"]
        info = read_json(Path(record["output_path"]) / "info.json")
        generated = len(info.get("predictions", []))
        if generated != record["sample_num"]:
            raise RuntimeError(f"仅生成 {generated}/{record['sample_num']} 个构象")
        with self.process_lock:
            if self.store.get(target_id)["status"] == "cancelled":
                return
            self.store.update(target_id, status="analyzing", stage=4, stage_label=STAGES[4], generated=generated)
        self.analysis(target_id, 10, allow_active=True)
        with self.process_lock:
            if self.store.get(target_id)["status"] != "cancelled":
                self.store.update(target_id, status="completed", stage=5, stage_label=STAGES[5],
                                  finished_at=now(), generated=generated, error=None, error_code=None, error_params={},
                                  elapsed_seconds=info.get("stage_times", {}).get("total", {}).get("seconds"))

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            records = [record for record in self.store.list() if record["status"] == "queued"]
            if not records:
                self.wake.wait(1)
                self.wake.clear()
                continue
            record = records[0]
            try:
                self._run(record)
            except Exception as exc:
                with self.process_lock:
                    current = self.store.get(record["id"], include_deleted=True)
                    if current["status"] != "cancelled" and not current.get("deleted_at"):
                        self.store.update(record["id"], status="failed", finished_at=now(), **error_fields(exc))

    def _run(self, record: dict) -> None:
        with self.target_lock(record["id"]):
            self._run_locked(record)

    def _run_locked(self, record: dict) -> None:
        target_id = record["id"]
        record = self.store.get(target_id)
        if record["status"] != "queued":
            return
        job_path = Path(record["job_path"])
        if self._prediction_ready(record):
            if self.store.get(target_id)["status"] == "queued":
                self._finish_analysis(record)
            return
        with self.process_lock:
            if self.stop_event.is_set() or self.store.get(target_id)["status"] != "queued":
                return
            self.store.update(target_id, status="running", stage=1, stage_label=STAGES[1], started_at=now(),
                              error=None, error_code=None, error_params={})
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"})
            command = [self.interpreter, "-u", "-m", "trflow", "predict", str(job_path / "request.json")]
            if self.env:
                command.extend(["--env", self.env])
            with (job_path / "run.log").open("w", encoding="utf-8") as log:
                self.process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=os.name != "nt")
            self.running_id = target_id
            process = self.process
        last_stage = 1
        while process.poll() is None:
            if self.stop_event.wait(0.6):
                self._terminate(process)
                break
            log_text = tail(job_path / "run.log")
            stages = re.findall(r"\[(\d)/3\]", log_text)
            stage = int(stages[-1]) if stages else last_stage
            if stage != last_stage and self.store.get(target_id)["status"] != "cancelled":
                self.store.update(target_id, stage=stage, stage_label=STAGES[stage])
                last_stage = stage
        return_code = process.wait()
        with self.process_lock:
            self.process = None
            self.running_id = None
        if self.store.get(target_id)["status"] == "cancelled":
            return
        if return_code != 0 or self.stop_event.is_set():
            if self.stop_event.is_set():
                raise WebRunError("运行中断", "run_interrupted")
            raise WebRunError(self.log(record)[-6000:] or f"预测进程退出：{return_code}",
                              "prediction_failed", exit_code=return_code)
        info_file = Path(record["output_path"]) / "info.json"
        if not info_file.is_file():
            raise RuntimeError("预测未生成 info.json，请查看运行日志")
        self._finish_analysis(record)

    def analysis(self, target_id: str, k: int = 10, allow_active: bool = False) -> dict:
        record = self.store.get(target_id)
        if record["status"] != "completed" and not (allow_active and record["status"] == "analyzing"):
            raise ValueError("结构尚未生成完成")
        if type(k) is not int or k < 1:
            raise ValueError("聚类数必须为正整数")
        with self.target_lock(target_id):
            record = self.store.get(target_id)
            if record["status"] != "completed" and not (allow_active and record["status"] == "analyzing"):
                raise ValueError("结构尚未生成完成")
            analysis_dir = Path(record["job_path"]) / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            result_file = analysis_dir / "response.json"
            # BLAS/OpenMP and GPU inference remain in independent processes.
            # A native numerical-library failure cannot terminate the HTTP server.
            command = [
                sys.executable, "-u", "-m", "trflow.web_analysis",
                "--predictions", str(Path(record["output_path"]) / "predictions"),
                "--analysis", str(analysis_dir),
                "--metadata", str(Path(record["output_path"]) / "info.json"),
                "--k", str(k), "--result", str(result_file),
            ]
            analysis_env = os.environ.copy()
            analysis_env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            with self.process_lock:
                if self.store.get(target_id)["status"] == "cancelled":
                    raise ValueError("任务已取消")
                process = subprocess.Popen(command, cwd=REPO, env=analysis_env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != "nt")
                self.analysis_processes[target_id] = process
            try:
                stdout, _ = process.communicate(timeout=1800)
                (analysis_dir / "analysis.log").write_bytes(stdout)
                if process.returncode != 0:
                    raise RuntimeError("结构分析失败：" + stdout.decode("utf-8", errors="replace")[-6000:])
                result = read_json(result_file)
            except subprocess.TimeoutExpired as exc:
                self._terminate(process)
                process.communicate()
                raise RuntimeError("结构分析超时，可从详情页重试") from exc
            finally:
                with self.process_lock:
                    self.analysis_processes.pop(target_id, None)
            for structure in result["structures"]:
                structure["url"] = f"/api/targets/{target_id}/structures/{structure['file']}?aligned=1"
                structure["original_url"] = f"/api/targets/{target_id}/structures/{structure['file']}"
            return result

    def archive(self, target_id: str, k: int = 10) -> bytes:
        if self.store.get(target_id)["status"] != "completed":
            raise ValueError("结构尚未生成完成")
        with self.target_lock(target_id):
            return self._archive(target_id, k)

    def _archive(self, target_id: str, k: int = 10) -> bytes:
        record = self.store.get(target_id)
        result = self.analysis(target_id, k)
        # Download paths are portable and independent of the internal analysis
        # cache. Never mutate the API/cache response or original prediction PDBs.
        exported = copy.deepcopy(result)
        analysis_dir = Path(record["job_path"]) / "analysis"
        aligned_paths = {}
        prediction_files = {}
        for structure in exported["structures"]:
            index = structure["index"]
            filename = structure["filename"]
            aligned_paths[index] = contained(analysis_dir, structure["aligned_file"])
            prediction_files[index] = f"prediction/{filename}"
            structure["file"] = prediction_files[index]
            structure["aligned_file"] = prediction_files[index]
            structure.pop("url", None)
            structure.pop("original_url", None)
        exported["reference"]["file"] = prediction_files[exported["reference"]["index"]]
        representative_files = {}
        for cluster in exported["clusters"]:
            representative = cluster["representative"]
            filename = Path(prediction_files[representative]).name
            representative_files[cluster["id"]] = f"clusters/representatives/cluster_{cluster['id']:03d}_{filename}"
            cluster["representative_file"] = representative_files[cluster["id"]]
            cluster["representative_prediction_file"] = prediction_files[representative]
            cluster["member_files"] = [prediction_files[index] for index in cluster["members"]]
        for point in exported["embedding"]:
            point["file"] = prediction_files[point["index"]]
        exported["target"] = {"id": target_id, "name": record["name"]}
        exported["export"] = {
            "prediction_directory": "prediction",
            "representatives_directory": "clusters/representatives",
            "structure_indices": "zero-based",
            "cluster_ids": "one-based",
            "coordinates": "All PDB coordinates use the same C-alpha Kabsch reference alignment.",
        }

        def csv_text(fields: list[str], rows: list[dict]) -> str:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            return stream.getvalue()

        assignment_fields = ["index", "file", "cluster", "is_representative", "rmsd_to_reference", "mean_plddt", "model"]
        assignments = [{
            "index": structure["index"], "file": structure["file"], "cluster": structure["cluster"],
            "is_representative": int(structure["representative"]),
            "rmsd_to_reference": structure["rmsd_to_reference"],
            "mean_plddt": structure.get("mean_plddt"), "model": structure.get("model"),
        } for structure in exported["structures"]]
        summary_fields = ["cluster", "size", "representative_index", "representative_file", "representative_prediction_file", "mean_rmsd", "max_rmsd"]
        summaries = [{
            "cluster": cluster["id"], "size": cluster["count"], "representative_index": cluster["representative"],
            "representative_file": cluster["representative_file"],
            "representative_prediction_file": cluster["representative_prediction_file"],
            "mean_rmsd": cluster["mean_rmsd"], "max_rmsd": cluster["max_rmsd"],
        } for cluster in exported["clusters"]]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            source = Path(record["output_path"])
            for index, filename in prediction_files.items():
                archive.write(aligned_paths[index], filename)
            for cluster in exported["clusters"]:
                archive.write(aligned_paths[cluster["representative"]], cluster["representative_file"])
            for path in source.glob("*.json"):
                archive.write(path, path.name)
            archive.writestr("clusters/clusters.json", json.dumps(exported, ensure_ascii=False, indent=2))
            archive.writestr("clusters/assignments.csv", csv_text(assignment_fields, assignments))
            archive.writestr("clusters/summary.csv", csv_text(summary_fields, summaries))
            archive.writestr("README.txt", (
                f"trFlow results\nTarget: {record['name']}\nID: {target_id}\n"
                f"Conformations: {exported['count']}\n"
                f"Clusters: {exported['k']} (requested: {exported['requested_k']})\n\n"
                "prediction/: All generated PDB structures, rigidly aligned to the first predicted structure.\n"
                "The C-alpha Kabsch fit is applied to all atom coordinates. No additional alignment is needed.\n"
                "clusters/representatives/: One aligned medoid conformation for each selected cluster.\n"
                "clusters/assignments.csv: Per-conformation cluster membership, RMSD, confidence and model.\n"
                "clusters/summary.csv: Cluster sizes, representative paths and within-cluster RMSD statistics.\n"
                "clusters/clusters.json: Complete selected clustering, members, distances, PCA and methods.\n"
                "Structure indices are zero-based; cluster IDs are one-based. RMSD values are in Angstrom.\n"
                "mean_plddt is normalized to 0-1; empty cells indicate unavailable values.\n"
                "The effective cluster count is capped by the number of generated conformations.\n"
                "info.json and other prediction JSON files retain the original run metadata.\n"
                "Original unaligned prediction files on the server are preserved.\n"
            ))
            log = self.log(record, full=True)
            if log.strip():
                archive.writestr("run.log", log)
        return buffer.getvalue()

    def close(self) -> None:
        self.stop_event.set()
        self.wake.set()
        with self.process_lock:
            if self.process:
                self._terminate(self.process)
            for process in list(self.analysis_processes.values()):
                self._terminate(process)
        if self.thread:
            self.thread.join(timeout=10)
        if not self.service_lock.closed:
            self.service_lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.service_lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.service_lock, fcntl.LOCK_UN)
            self.service_lock.close()


def create_handler(manager: JobManager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "trFlowLocal/0.1"

        def log_message(self, fmt, *args):
            # Polling should not fill a terminal with access logs.
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                print(fmt % args, file=sys.stderr)

        def _send(self, body, status=200, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            if filename:
                disposition = (f'attachment; filename="{filename}"' if filename.isascii() else
                               f'attachment; filename="trflow_target.zip"; filename*=UTF-8\'\'{quote(filename, safe="")}')
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(body)

        def _check_host(self):
            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"127.0.0.1", "localhost", "[::1]"}:
                raise ValueError("请通过 localhost 或 127.0.0.1 访问")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                raise ValueError("不允许来自其他网页的跨域请求")

        def _json_body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("请求为空或超过 64 MB；请减少批量上传文件大小") from exc
            if not 0 < length <= MAX_BODY:
                raise ValueError("请求为空或超过 64 MB；请减少批量上传文件大小")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("提交内容须为 application/json")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("请求须为 JSON 对象")
            return body

        @staticmethod
        def _cluster_count(query):
            try:
                k = int(query.get("k", ["10"])[0])
            except (ValueError, TypeError) as exc:
                raise ValueError("聚类数必须为正整数") from exc
            if k < 1:
                raise ValueError("聚类数必须为正整数")
            return k

        def _route(self, method):
            self._check_host()
            url = urlsplit(self.path)
            path = unquote(url.path)
            query = parse_qs(url.query)
            if method == "GET" and path == "/api/config":
                return self._send(manager.config())
            if method == "GET" and path == "/api/example":
                return self._send(example_target())
            if path == "/api/targets":
                if method == "GET":
                    return self._send({"targets": [manager.public(record) for record in reversed(manager.store.list())]})
                if method == "POST":
                    return self._send({"targets": manager.submit(self._json_body().get("targets"))}, 201)
            if method == "POST" and path == "/api/import":
                return self._send({"targets": manager.import_config(self._json_body())}, 201)
            if method == "POST" and path == "/api/example":
                payload = self._json_body()
                return self._send({"targets": manager.submit([{
                    **example_target(),
                    "options": payload.get("options", {}),
                }])}, 201)
            if method == "GET" and path == "/api/recycle-bin":
                return self._send({"targets": manager.recycled()})
            if method == "POST" and path.startswith("/api/recycle-bin/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "restore":
                    return self._send({"target": manager.restore(parts[2])})
            if path.startswith("/api/targets/"):
                parts = path.strip("/").split("/")
                if len(parts) < 3:
                    raise FileNotFoundError("目标不存在")
                target_id = parts[2]
                if method == "DELETE" and len(parts) == 3:
                    return self._send(manager.delete(target_id))
                record = manager.store.get(target_id)
                if method == "GET" and len(parts) == 3:
                    return self._send(manager.public(record, logs=True))
                if len(parts) == 4:
                    action = parts[3]
                    if method == "POST" and action == "cancel":
                        return self._send(manager.cancel(target_id))
                    if method == "POST" and action == "retry":
                        return self._send({"targets": manager.retry(target_id)}, 201)
                    if method == "GET" and action == "analysis":
                        return self._send(manager.analysis(target_id, self._cluster_count(query)))
                    if method == "GET" and action == "log":
                        return self._send(manager.log(record, full=True).encode("utf-8"), content_type="text/plain; charset=utf-8")
                    if method == "GET" and action == "download":
                        return self._send(manager.archive(target_id, self._cluster_count(query)),
                                          content_type="application/zip", filename=download_filename(record))
                if method == "GET" and len(parts) == 5 and parts[3] == "structures":
                    with manager.target_lock(target_id):
                        record = manager.store.get(target_id)
                        root = (Path(record["job_path"]) / "analysis" / "aligned" if query.get("aligned") == ["1"]
                                else Path(record["output_path"]) / "predictions")
                        filename = parts[4]
                        if "/" in filename or "\\" in filename or not filename.lower().endswith(".pdb"):
                            raise ValueError("结构文件名称无效")
                        file = contained(root, filename)
                        if not file.is_file():
                            raise FileNotFoundError("结构文件不存在")
                        data = file.read_bytes()
                    return self._send(data, content_type="chemical/x-pdb")
            if method == "GET":
                if path.startswith("/static/"):
                    file = contained(STATIC, path[len("/static/"):])
                elif path == "/" or path.startswith("/targets/"):
                    file = STATIC / "index.html"
                else:
                    raise FileNotFoundError("页面不存在")
                if not file.is_file():
                    raise FileNotFoundError("文件不存在")
                # Windows MIME registry entries may incorrectly classify JS/CSS.
                # Serve known assets consistently on both supported platforms.
                content_type = {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".js": "text/javascript; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(file.suffix.lower()) or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                return self._send(file.read_bytes(), content_type=content_type)
            raise FileNotFoundError("接口不存在")

        def _handle(self, method):
            try:
                self._route(method)
            except TargetConflictError as exc:
                self._send({**error_fields(exc), "code": exc.code}, 409)
            except (KeyError, FileNotFoundError) as exc:
                self._send(error_fields(exc), 404)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(error_fields(exc), 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send(error_fields(exc), 500)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_DELETE(self):
            self._handle("DELETE")

    return Handler


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8765, help="local web server port")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="persistent queue and result directory")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for trFlow predictions")
    parser.add_argument("--env", default=None, help="trFlow environment configuration")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    parser.add_argument("--no-import-existing", action="store_true", help="do not show existing outputs in the target list")


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        raise ValueError("port 必须为 1 至 65535")
    # Bind before creating a worker: a second server cannot run queued GPU jobs.
    server = LocalHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    try:
        manager = JobManager(Path(args.data_dir), args.python, args.env, not args.no_import_existing)
    except Exception:
        server.server_close()
        raise
    server.RequestHandlerClass = create_handler(manager)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"trFlow local workbench: {url}\nQueue and results: {manager.data_dir}\n"
          "One target runs at a time. Press Ctrl+C to stop the service.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        server.server_close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start the local trFlow web interface.")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
