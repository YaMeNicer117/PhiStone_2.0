"""将本次生成的条件 ``.pt`` 同步到两张完整标准词表。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    import utils
    import PhiSSeparator_atom_encoder_train as atom_train
    import PhiSSeparator_fragment_encoder_train as fragment_train
    from PhiSSeparator_atom_encoder_resolver import (
        AtomEmbeddingResolver,
    )
    from PhiSSeparator_fragment_encoder_resolver import (
        FragmentEmbeddingResolver,
    )
except ImportError:  # 支持作为 PhiSSeparator 子模块导入
    from . import utils
    from . import PhiSSeparator_atom_encoder_train as atom_train
    from . import PhiSSeparator_fragment_encoder_train as fragment_train
    from .PhiSSeparator_atom_encoder_resolver import (
        AtomEmbeddingResolver,
    )
    from .PhiSSeparator_fragment_encoder_resolver import (
        FragmentEmbeddingResolver,
    )


GLOBAL_SMILES_TOKEN = "<GLOBAL>"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _torch_load_condition(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # 兼容旧版 PyTorch
        return torch.load(path, map_location="cpu")


def _condition_fragment_smiles(payload, path):
    if isinstance(payload, Mapping):
        if "fragment_smiles" not in payload:
            raise ValueError(f"条件文件缺少 fragment_smiles: {path}")
        values = payload["fragment_smiles"]
    elif hasattr(payload, "fragment_smiles"):
        values = payload.fragment_smiles
    else:
        raise ValueError(f"条件文件缺少 fragment_smiles: {path}")
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"fragment_smiles 必须是逐节点字符串序列: {path}"
        )
    try:
        values = list(values)
    except TypeError as exc:
        raise TypeError(
            f"fragment_smiles 必须是逐节点字符串序列: {path}"
        ) from exc
    return values


def _normalize_exact_smiles(value, *, path, node_index):
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"fragment_smiles 不是 UTF-8: {path}, node={node_index}"
            ) from exc
    if not isinstance(value, str) or not value:
        raise TypeError(
            "fragment_smiles 必须是非空字符串: "
            f"{path}, node={node_index}"
        )
    return value


def _canonical_smiles_or_raise(smiles, *, path, node_index):
    if smiles == GLOBAL_SMILES_TOKEN:
        return None
    if (
        smiles in utils.RAW_FRAGMENT_SPECIAL_TOKENS
        or (smiles.startswith("<") and smiles.endswith(">"))
    ):
        raise ValueError(
            "条件文件包含不允许同步的特殊标记: "
            f"{smiles!r}, {path}, node={node_index}"
        )
    canonical_smiles, _ = utils.canonicalize_fragment_smiles(smiles)
    if canonical_smiles != smiles:
        raise ValueError(
            "条件文件中的 SMILES 不是稳定规范形式，拒绝改写 .pt: "
            f"{smiles!r} != {canonical_smiles!r}, "
            f"{path}, node={node_index}"
        )
    return canonical_smiles


def _resolver_report(resolver):
    resolution = resolver.last_resolution
    sources = list(resolution["source"])
    return {
        "requested_count": int(resolution["requested_count"]),
        "cache_hits": int(sum(source == "cache" for source in sources)),
        "checkpoint_results": int(
            sum(source == "checkpoint" for source in sources)
        ),
        "newly_added_to_standard_table": list(
            resolution["newly_added_to_standard_table"]
        ),
        "newly_added_to_raw_corpus": list(
            resolution["newly_added_to_raw_corpus"]
        ),
    }


def _write_report(report_path, report):
    if report_path is None:
        return
    utils.atomic_write_json(report_path, report)


def synchronize_condition_vocabularies(
    condition_paths,
    *,
    device="auto",
    report_path=None,
) -> dict:
    """同步显式给出的条件文件；不会扫描其所在目录。"""
    if isinstance(condition_paths, (str, bytes, Path)):
        condition_paths = [condition_paths]
    else:
        condition_paths = list(condition_paths)

    normalized_paths = []
    seen_paths = set()
    for raw_path in condition_paths:
        path = Path(raw_path).expanduser().resolve()
        path_key = str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        normalized_paths.append(path)

    resolved_report_path = (
        None
        if report_path is None
        else Path(report_path).expanduser().resolve()
    )
    report = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "device": str(device),
        "condition_paths": [str(path) for path in normalized_paths],
        "condition_file_count": len(normalized_paths),
        "report_path": (
            None
            if resolved_report_path is None
            else str(resolved_report_path)
        ),
        "stage": "validate_condition_paths",
    }

    try:
        if not normalized_paths:
            raise ValueError("condition_paths 至少需要一个 .pt 文件")

        unique_smiles = []
        seen_smiles = set()
        condition_details = []
        non_global_occurrences = 0
        report["stage"] = "read_condition_files"
        for path in normalized_paths:
            if path.suffix.lower() != ".pt":
                raise ValueError(f"条件文件必须使用 .pt 扩展名: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"条件文件不存在: {path}")
            payload = _torch_load_condition(path)
            values = _condition_fragment_smiles(payload, path)
            local_non_global = 0
            for node_index, value in enumerate(values):
                exact_smiles = _normalize_exact_smiles(
                    value, path=path, node_index=node_index
                )
                canonical_smiles = _canonical_smiles_or_raise(
                    exact_smiles, path=path, node_index=node_index
                )
                if canonical_smiles is None:
                    continue
                local_non_global += 1
                non_global_occurrences += 1
                if canonical_smiles not in seen_smiles:
                    seen_smiles.add(canonical_smiles)
                    unique_smiles.append(canonical_smiles)
            condition_details.append({
                "path": str(path),
                "node_count": len(values),
                "non_global_smiles_count": local_non_global,
            })
            del payload, values

        report["conditions"] = condition_details
        report["non_global_smiles_occurrence_count"] = (
            non_global_occurrences
        )
        report["unique_smiles_count"] = len(unique_smiles)

        report["stage"] = "resolve_fragment_vocabulary"
        fragment_resolver = FragmentEmbeddingResolver(
            device_spec=device
        )
        fragment_resolver.resolve_many(unique_smiles)
        if fragment_resolver.last_resolution[
            "canonical_smiles"
        ] != unique_smiles:
            raise RuntimeError(
                "片段 Resolver 返回的规范 SMILES 与条件文件不一致"
            )
        report["fragment"] = _resolver_report(fragment_resolver)
        fragment_table_path = fragment_resolver.embedding_table_path
        fragment_metadata_path = (
            fragment_resolver.embedding_metadata_path
        )
        del fragment_resolver

        report["stage"] = "resolve_atom_vocabulary"
        atom_resolver = AtomEmbeddingResolver(device=device)
        atom_results = atom_resolver.resolve_many(unique_smiles)
        if [
            result["canonical_smiles"] for result in atom_results
        ] != unique_smiles:
            raise RuntimeError(
                "原子 Resolver 返回的规范 SMILES 与条件文件不一致"
            )
        del atom_results
        report["atom"] = _resolver_report(atom_resolver)
        report["atom"]["compatibility_anchors_path"] = (
            atom_resolver.compatibility_anchors_path
        )
        atom_table_path = atom_resolver.embedding_table_path
        atom_metadata_path = atom_resolver.embedding_metadata_path
        del atom_resolver

        report["stage"] = "verify_authoritative_tables"
        fragment_arrays, fragment_metadata = (
            fragment_train.load_fragment_embedding_table(
                fragment_table_path,
                fragment_metadata_path,
            )
        )
        atom_arrays, atom_metadata = atom_train.load_atom_embedding_table(
            atom_table_path,
            atom_metadata_path,
        )
        fragment_values = set(
            fragment_arrays["smiles"].astype(str).tolist()
        )
        atom_values = set(atom_arrays["smiles"].astype(str).tolist())
        missing_fragment = [
            smiles for smiles in unique_smiles
            if smiles not in fragment_values
        ]
        missing_atom = [
            smiles for smiles in unique_smiles if smiles not in atom_values
        ]
        if missing_fragment or missing_atom:
            raise RuntimeError(
                "词汇同步后的权威表复核失败: "
                f"fragment_missing={missing_fragment[:20]}, "
                f"atom_missing={missing_atom[:20]}"
            )
        report["verification"] = {
            "fragment_table_path": (
                fragment_table_path
            ),
            "fragment_num_smiles": int(
                fragment_metadata["num_smiles"]
            ),
            "fragment_table_sha256": str(
                fragment_metadata["table_sha256"]
            ),
            "atom_table_path": atom_table_path,
            "atom_num_smiles": int(atom_metadata["num_smiles"]),
            "atom_num_atoms": int(atom_metadata["num_atoms"]),
            "atom_table_sha256": str(atom_metadata["table_sha256"]),
        }
        report["status"] = "complete"
        report["stage"] = "complete"
        report["finished_at_utc"] = _utc_now()
        _write_report(resolved_report_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["finished_at_utc"] = _utc_now()
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        try:
            _write_report(resolved_report_path, report)
        except Exception as report_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "写入词汇同步失败报告时再次出错: "
                    f"{type(report_exc).__name__}: {report_exc}"
                )
        raise
