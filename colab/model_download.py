"""Public Colab weight downloads; no Drive mount or model deserialization.

The notebook embeds this file verbatim so its pinned model checkout need not
change. Only the four reviewed files are requested, not arbitrary folder data.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid


GDOWN_VERSION = "6.4.2"
PUBLIC_FOLDER_ID = "17dy2aKS3f16_SulYdWwx_6wujKWPODH_"
PUBLIC_FOLDER_URL = "https://drive.google.com/drive/folders/" + PUBLIC_FOLDER_ID
MODEL_FILES = {
    "trflow_xray.pth": {
        "id": "1kVrCOUDcpbhVBdx1E8W6StkLeCnNFrhg", "bytes": 119752653,
        "sha256": "ef48e7b60d7621eb62ceded9d762528ab5a9b384e4963c72f71afa120442ff53",
    },
    "trflow_nmr.pth": {
        "id": "1CbKfwMh-mvmcAkqAwJIpTGge5scmTNnX", "bytes": 119749792,
        "sha256": "5774ec7bd4e5297a593c49f35e9af8947329897b925c96c183b4f4d23aa7f189",
    },
    "esm_msa1_t12_100M_UR50S.pt": {
        "id": "1Skhc9fIv4U6risVWRkhWHz5-r7D4JE5E", "bytes": 1378199982,
        "sha256": "cd932794cc2f23e179c7335a6f95a45199ba38a5f4b664976e77c8f71b2e6878",
    },
    "openfold_params_model_5_ptm.npz": {
        "id": "1Wxw52VtIeEhqfz-OYtF7H5q0B_wDF7hk", "bytes": 371746284,
        "sha256": "2f8d0bd193e7d6e837958ba92acdc3f02b23b61bf841fa043fc076ee0e1996a5",
    },
}
DISK_RESERVE = 512 * 1024**2
FILE_TIMEOUT = 1800


class ModelCacheError(ValueError):
    """Unsafe cache paths must not be used, even through a fallback."""


class ModelDownloadError(RuntimeError):
    """A download or its pinned integrity check failed."""


def _safe_directory(path):
    path = Path(path).expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ModelCacheError("The model cache must not contain linked directories.")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ModelCacheError("The model cache path is not a directory.")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path, expected):
    if path.is_symlink():
        raise ModelCacheError("A model file must not be a link: " + path.name)
    if path.exists() and not path.is_file():
        raise ModelCacheError("A model file path is not a regular file: " + path.name)
    return (path.is_file() and path.stat().st_size == expected["bytes"]
            and _sha256(path) == expected["sha256"])


def validate_weights(directory):
    """Validate names, lengths and SHA-256 without importing model loaders."""
    directory = Path(directory).expanduser().absolute()
    for component in (directory, *directory.parents):
        if component.is_symlink():
            raise ModelCacheError("The model folder must not be a link.")
    if not directory.is_dir():
        raise ModelDownloadError("The model folder does not exist.")
    if set(p.name for p in directory.iterdir()) != set(MODEL_FILES):
        raise ModelDownloadError("The model folder must contain exactly four checkpoints.")
    for name, expected in MODEL_FILES.items():
        if not _valid_file(directory / name, expected):
            raise ModelDownloadError("Model integrity check failed: " + name)
    return directory


def _record(cache, directory, source):
    receipt = cache / "current.json"
    if receipt.is_symlink():
        raise ModelCacheError("The model-cache record must not be a link.")
    record = {
        "schema_version": 1, "source": source, "folder_url": PUBLIC_FOLDER_URL,
        "directory": str(directory), "files": MODEL_FILES,
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache,
                                     prefix=".receipt-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)


def _default_downloader(**kwargs):
    # Imported only for a cold cache. No browser cookies or Drive authorization.
    import gdown
    from importlib.metadata import version
    if version("gdown") != GDOWN_VERSION or gdown.__version__ != GDOWN_VERSION:
        raise ModelDownloadError("Rerun Step 1 to install the notebook's download tool.")
    return gdown.download(**kwargs)


def _download_one(name, expected, incoming, progress, completed_bytes, total_bytes,
                  downloader):
    output = incoming / (name + ".download")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ModelCacheError("Unsafe partial model-file path: " + name)
    for partial in incoming.glob(output.name + "*.part"):
        if partial.is_symlink() or not partial.is_file():
            raise ModelCacheError("Unsafe partial model-file path: " + name)
    # gdown skips existing output files with resume=True. Never skip a bad one.
    if output.exists() and not _valid_file(output, expected):
        os.replace(output, incoming / (name + ".invalid-" + uuid.uuid4().hex))
    started = time.monotonic()
    last_update = -1.0

    def report(received, announced):
        nonlocal last_update
        if not isinstance(received, int) or received < 0 or received > expected["bytes"]:
            raise ModelDownloadError("Unexpected model download size: " + name)
        if announced is not None and announced != expected["bytes"]:
            raise ModelDownloadError("The model download size differs from the reviewed file: " + name)
        now = time.monotonic()
        if now - started > FILE_TIMEOUT:
            raise ModelDownloadError("The Drive download timed out: " + name)
        if now - last_update >= 1 or received == expected["bytes"]:
            if shutil.disk_usage(incoming).free < DISK_RESERVE:
                raise ModelDownloadError("Not enough runtime storage for model files.")
            if progress is not None:
                percent = 100 * (completed_bytes + received) / total_bytes
                progress(f"Downloading model files · {percent:.0f}% · {name}")
            last_update = now

    try:
        result = downloader(id=expected["id"], output=str(output), quiet=True,
                            use_cookies=False, verify=True, resume=True,
                            timeout=(15, 60), retries=2, progress=report)
    except (KeyboardInterrupt, ModelCacheError):
        raise
    except Exception as error:
        raise ModelDownloadError("Drive download could not finish: " + name) from error
    if result is None or not _valid_file(output, expected):
        if output.is_file() and not output.is_symlink():
            os.replace(output, incoming / (name + ".invalid-" + uuid.uuid4().hex))
        raise ModelDownloadError("Model integrity check failed after download: " + name)
    return output


def ensure_model_weights(cache_dir, progress=None, *, legacy_directory=None,
                         fallback=None, downloader=None):
    """Use verified runtime files, then public Drive, then an optional author server.

    Partial transfers stay in this runtime for gdown's range-resume support.
    Completed files are reused, and no folder is returned before all four pass.
    KeyboardInterrupt always cancels; it must not trigger a fresh server download.
    """
    cache = _safe_directory(cache_dir)
    if (cache / "current.json").is_symlink():
        raise ModelCacheError("The model-cache record must not be a link.")
    identity = hashlib.sha256(json.dumps(MODEL_FILES, sort_keys=True).encode()).hexdigest()[:16]
    models = _safe_directory(cache / ("models-" + identity))
    if any(p.name not in MODEL_FILES for p in models.iterdir()):
        raise ModelCacheError("Unexpected files in the model cache; use a new runtime folder.")
    if progress is not None:
        progress("Checking saved model files")
    valid = {name for name, expected in MODEL_FILES.items()
             if _valid_file(models / name, expected)}
    if len(valid) == len(MODEL_FILES):
        if progress is not None:
            progress("Using saved model files")
        _record(cache, models, "Google Drive")
        return models
    if legacy_directory is not None:
        try:
            legacy = validate_weights(legacy_directory)
        except ModelDownloadError:
            pass
        else:
            if progress is not None:
                progress("Using saved model files")
            _record(cache, legacy, "Existing runtime cache")
            return legacy
    missing_bytes = sum(row["bytes"] for name, row in MODEL_FILES.items() if name not in valid)
    if shutil.disk_usage(cache).free < missing_bytes + DISK_RESERVE:
        raise ModelDownloadError("Not enough runtime storage; free space and rerun Step 1.")
    incoming = _safe_directory(cache / ("partial-" + identity))
    completed_bytes = sum(MODEL_FILES[name]["bytes"] for name in valid)
    total_bytes = sum(row["bytes"] for row in MODEL_FILES.values())
    try:
        for name, expected in MODEL_FILES.items():
            if name in valid:
                continue
            pending = _download_one(name, expected, incoming, progress, completed_bytes,
                                    total_bytes, downloader or _default_downloader)
            os.replace(pending, models / name)
            completed_bytes += expected["bytes"]
    except ModelDownloadError:
        if fallback is None:
            raise
        if progress is not None:
            progress("Drive download unavailable; trying the author server")
        try:
            alternative = validate_weights(fallback())
        except (KeyboardInterrupt, ModelCacheError):
            raise
        except Exception as fallback_error:
            raise ModelDownloadError("Model download could not finish. Rerun Step 1; saved Drive transfers will resume.") from fallback_error
        _record(cache, alternative, "Author server")
        return alternative
    _record(cache, models, "Google Drive")
    if progress is not None:
        progress("Model files ready")
    return models
