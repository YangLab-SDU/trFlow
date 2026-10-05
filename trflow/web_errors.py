"""Stable web error codes, independent of the user's interface language.

The original diagnostic is always retained. Only exact application messages
are recognized, so a model traceback containing the same words is not rewritten.
"""
from __future__ import annotations

import json
import re


_MESSAGES = {
    "服务已中断；可重新提交此目标。": "service_interrupted",
    "运行中断": "run_interrupted",
    "目标不存在": "target_not_found",
    "结构尚未生成完成": "structures_not_ready",
    "任务已取消": "target_cancelled",
    "结构分析超时，可从详情页重试": "analysis_timeout",
    "预测未生成 info.json，请查看运行日志": "prediction_info_missing",
    "路径必须位于当前任务目录内": "path_outside_target",
    "此数据目录已有 trFlow 网页服务运行，请使用现有服务。": "service_already_running",
    "options 必须是 JSON 对象": "options_object",
    "models 请选择 Xray、NMR 或两者": "invalid_models",
    "seed 为空或为 0 至 4294967295 之间的整数": "invalid_seed",
    "gpus 必须为 GPU 编号数组": "invalid_gpus",
    "max_workers 必须为正整数或空": "invalid_max_workers",
    "每个 target 必须是 JSON 对象": "target_object",
    "目标名称必须为 1 至 120 个字符": "invalid_target_name",
    "一次可提交 1 至 50 个 target": "invalid_target_count",
    "JSON 需包含 samples 数组；可使用 trFlow 原有输入格式": "config_samples_required",
    "files 必须是数组": "files_array",
    "上传文件需包含 name 与 content": "upload_fields_required",
    "samples 中每个元素必须为对象": "sample_object",
    "Invalid target identifier": "invalid_target_id",
    "The target directory is outside the workbench": "target_directory_outside",
    "The output directory is outside this target": "output_directory_outside",
    "The recycle bin is outside the workbench": "recycle_directory_outside",
    "The recycle-bin metadata does not match this target": "recycle_metadata_mismatch",
    "The recycled target directory is missing": "recycled_directory_missing",
    "只有等待或运行中的目标可以取消": "target_not_active",
    "目标仍在等待或运行中": "target_still_active",
    "已有结果无法重试，请从 MSA 重新提交": "existing_result_not_retryable",
    "聚类数必须为正整数": "invalid_cluster_count",
    "请通过 localhost 或 127.0.0.1 访问": "localhost_required",
    "不允许来自其他网页的跨域请求": "cross_origin_denied",
    "请求为空或超过 64 MB；请减少批量上传文件大小": "invalid_body_size",
    "提交内容须为 application/json": "json_content_required",
    "请求须为 JSON 对象": "json_object_required",
    "结构文件名称无效": "invalid_structure_filename",
    "结构文件不存在": "structure_not_found",
    "页面不存在": "page_not_found",
    "文件不存在": "file_not_found",
    "接口不存在": "endpoint_not_found",
    "port 必须为 1 至 65535": "invalid_port",
    "Cancel this target before deleting it": "target_active",
    "The target is still releasing resources; try again shortly": "target_busy",
    "A recycle-bin entry already exists for this target": "recycle_conflict",
    "This target is not in the recycle bin": "not_deleted",
    "Restoring would overwrite an existing target directory": "restore_conflict",
}

_PATTERNS = (
    (r"仅生成 (?P<generated>\d+)/(?P<expected>\d+) 个构象", "incomplete_predictions"),
    (r"预测进程退出：(?P<exit_code>-?\d+)", "prediction_failed"),
    (r"结构分析失败：(?P<detail>[\s\S]*)", "analysis_failed"),
    (r"未知参数：(?P<fields>[^\n]+)", "unknown_options"),
    (r"(?P<field>sample_num|steps) 必须为有效正整数", "invalid_positive_integer"),
    (r"(?P<field>geometric_exploration|single_step|random_step|random_step_size|parallel|save_repr_npz) 必须为 true 或 false", "invalid_boolean"),
    (r"(?P<name>[^\n]+) 缺少 A3M 内容", "msa_missing"),
    (r"(?P<name>[^\n]+) 的 A3M 首条记录标题为空", "msa_header_empty"),
    (r"(?P<name>[^\n]+) 的 A3M 必须以 > 标题开头", "msa_header_required"),
    (r"(?P<name>[^\n]+) 的 A3M 首条记录必须为无 gap、无小写插入的目标序列", "msa_query_invalid"),
    (r"(?P<name>[^\n]+) 的 FASTA 格式不正确", "fasta_invalid"),
    (r"(?P<name>[^\n]+) 的 FASTA 与 A3M 首条序列不一致", "fasta_mismatch"),
    (r"(?P<name>[^\n]+) 的初始 PDB 缺少 ATOM 记录", "pdb_atoms_missing"),
    (r"上传文件名称重复：(?P<filename>[^\n]+)", "duplicate_upload"),
    (r"(?P<name>[^\n]+) 缺少 (?P<field>msa_path|fasta_path|init_pdb)", "sample_field_missing"),
    (r"本地路径须位于项目目录内；或将文件一并上传：(?P<path>[^\n]+)", "input_path_outside_project"),
    (r"找不到 (?P<path>[^\n]+)；请在页面上传相应文件", "input_file_missing"),
)


class WebRunError(RuntimeError):
    """A run failure with an original diagnostic and a stable presentation code."""

    def __init__(self, message: str, code: str, **params):
        super().__init__(message)
        self.error_code = code
        self.error_params = params


def error_fields(error, code: str | None = None, params: dict | None = None) -> dict:
    """Build serializable error fields without translating or mutating input."""
    raw = str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)
    code = code or getattr(error, "error_code", None) or getattr(error, "code", None)
    params = params if isinstance(params, dict) else getattr(error, "error_params", {})
    if not isinstance(params, dict):
        params = {}
    if not code and isinstance(error, json.JSONDecodeError):
        code, params = "invalid_json", {"detail": str(error)}
    if not code:
        code = _MESSAGES.get(raw)
    # Single-line exceptions can contain validation phrases too. A diagnostic
    # prefix is not a target name; explicit codes above still take precedence.
    diagnostic = re.match(r"^(?:[\w.]*?(?:Error|Exception|Warning)\s*:|Traceback\b)", raw.lstrip())
    if not code and diagnostic:
        return {"error": raw, "error_code": None, "error_params": dict(params)}
    if not code:
        for pattern, candidate in _PATTERNS:
            match = re.fullmatch(pattern, raw)
            if match:
                code, params = candidate, match.groupdict()
                for field in ("generated", "expected", "exit_code"):
                    if field in params:
                        params[field] = int(params[field])
                break
    return {"error": raw, "error_code": code, "error_params": dict(params)}
