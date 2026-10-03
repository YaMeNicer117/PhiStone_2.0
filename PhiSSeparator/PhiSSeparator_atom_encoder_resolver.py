import argparse
import json
import os
import socket
import threading
import time
import warnings

import numpy as np
import torch

try:
    import utils
    import PhiSSeparator_atom_encoder_train as atom_train
except ImportError:  # 支持作为 PhiSSeparator 子模块导入
    from . import utils
    from . import PhiSSeparator_atom_encoder_train as atom_train


DEFAULT_RAW_CORPUS = atom_train.DEFAULT_RAW_CORPUS
DEFAULT_RAW_METADATA = atom_train.DEFAULT_RAW_METADATA
# 复用训练脚本的权威路径，并防止默认权重扩展名再次发生漂移。
DEFAULT_CHECKPOINT = atom_train.DEFAULT_CHECKPOINT
if os.path.splitext(DEFAULT_CHECKPOINT)[1].lower() != ".pth":
    raise RuntimeError(
        "原子编码器训练脚本与 resolver 的默认 checkpoint "
    )
DEFAULT_EMBEDDING_TABLE = atom_train.DEFAULT_EMBEDDING_TABLE
DEFAULT_EMBEDDING_JSON = atom_train.DEFAULT_EMBEDDING_JSON
DEFAULT_EMBEDDING_METADATA = atom_train.DEFAULT_EMBEDDING_METADATA
DEFAULT_LOCK_FILE = os.path.join(
    atom_train.SHARED_DATA_DIR,
    "atom_encoder_resolver.lock",
)
DEFAULT_TRANSACTION_FILE = os.path.join(
    atom_train.SHARED_DATA_DIR,
    "atom_encoder_update_transaction.json",
)


class AtomEmbeddingResolver:
    """缓存优先、可并发追加的 64 维片段原子词嵌入解析器。"""

    def __init__(
        self,
        raw_corpus_path=DEFAULT_RAW_CORPUS,
        raw_metadata_path=DEFAULT_RAW_METADATA,
        checkpoint_path=DEFAULT_CHECKPOINT,
        embedding_table_path=DEFAULT_EMBEDDING_TABLE,
        embedding_metadata_path=DEFAULT_EMBEDDING_METADATA,
        lock_path=DEFAULT_LOCK_FILE,
        transaction_path=DEFAULT_TRANSACTION_FILE,
        compatibility_anchors_path=None,
        device="auto",
        lock_timeout_seconds=120.0,
        stale_lock_seconds=600.0,
    ):
        self.raw_corpus_path = os.path.abspath(
            os.fspath(raw_corpus_path)
        )
        self.raw_metadata_path = os.path.abspath(
            os.fspath(raw_metadata_path)
        )
        self.checkpoint_path = os.path.abspath(
            os.fspath(checkpoint_path)
        )
        self.embedding_table_path = os.path.abspath(
            os.fspath(embedding_table_path)
        )
        self.embedding_json_path = (
            utils.derive_embedding_json_mirror_path(
                self.embedding_table_path
            )
        )
        self.embedding_metadata_path = os.path.abspath(
            os.fspath(embedding_metadata_path)
        )
        self.lock_path = os.path.abspath(os.fspath(lock_path))
        self.transaction_path = os.path.abspath(
            os.fspath(transaction_path)
        )
        if compatibility_anchors_path is None:
            compatibility_anchors_path = (
                os.path.splitext(self.embedding_table_path)[0]
                + "_compatibility_anchors.json"
            )
        self.compatibility_anchors_path = os.path.abspath(
            os.fspath(compatibility_anchors_path)
        )
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.device = atom_train.resolve_device(device)
        self._thread_lock = threading.Lock()
        self.last_resolution = None

        required_paths = (
            self.raw_corpus_path,
            self.raw_metadata_path,
            self.checkpoint_path,
            self.embedding_table_path,
            self.embedding_metadata_path,
        )
        missing_paths = [
            path for path in required_paths if not os.path.isfile(path)
        ]
        if missing_paths:
            raise FileNotFoundError(
                "原子编码器产物不完整，请先运行 "
                "PhiSSeparator_atom_encoder_train.py："
                f"{missing_paths}"
            )

        self.checkpoint_sha256 = atom_train.file_sha256(
            self.checkpoint_path
        )
        self.model, self.checkpoint = (
            atom_train.load_atom_encoder_checkpoint(
                self.checkpoint_path, self.device
            )
        )
        self.observed_atomic_numbers = {
            int(value)
            for value in self.checkpoint["observed_atomic_numbers"]
        }

        try:
            self._reload_state()
        except (OSError, ValueError, json.JSONDecodeError):
            if not os.path.isfile(self.transaction_path):
                raise
            with self._new_lock():
                self._validate_transaction(self._read_transaction())
                self._repair_atomic_pairs_locked()
                self._reload_state()
                self._validate_checkpoint_compatibility()
                self._recover_locked()
                self._reload_state()

        self._validate_checkpoint_compatibility()
        missing_raw_smiles = (
            self._raw_smiles_missing_from_embedding_table()
        )
        raw_metadata_changed = self.embedding_metadata.get(
            "current_raw_corpus_sha256"
        ) != self.raw_metadata.get("corpus_sha256")
        if (
            os.path.isfile(self.transaction_path)
            or missing_raw_smiles
            or raw_metadata_changed
        ):
            with self._new_lock():
                self._recover_locked()
                self._reload_state()
                self._validate_checkpoint_compatibility()
        self._require_fully_synchronized()
        self._ensure_json_mirror()

    @staticmethod
    def _load_with_retry(loader, *paths):
        last_error = None
        for attempt in range(6):
            try:
                return loader(*paths)
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(0.05)
        raise last_error

    def _reload_state(self):
        self.raw_corpus, self.raw_metadata = self._load_with_retry(
            utils.load_raw_fragment_atom_corpus,
            self.raw_corpus_path,
            self.raw_metadata_path,
        )
        (
            self.embedding_table,
            self.embedding_metadata,
        ) = self._load_with_retry(
            atom_train.load_atom_embedding_table,
            self.embedding_table_path,
            self.embedding_metadata_path,
        )
        self.raw_index = {
            str(smiles): index
            for index, smiles in enumerate(self.raw_corpus["smiles"])
        }
        self.embedding_index = {
            str(smiles): index
            for index, smiles in enumerate(
                self.embedding_table["smiles"]
            )
        }

    def _new_lock(self):
        return utils.AtomicFileLock(
            self.lock_path,
            timeout_seconds=self.lock_timeout_seconds,
            stale_seconds=self.stale_lock_seconds,
        )

    def _ensure_json_mirror(self):
        """缺失或明显过期时，由权威 NPZ 在写锁内重建 JSON 镜像。"""
        if not utils.embedding_json_mirror_needs_refresh(
            self.embedding_json_path,
            self.embedding_table,
            self.embedding_metadata,
            table_kind="atom",
        ):
            return
        with self._new_lock():
            self._reload_state()
            if utils.embedding_json_mirror_needs_refresh(
                self.embedding_json_path,
                self.embedding_table,
                self.embedding_metadata,
                table_kind="atom",
            ):
                atom_train.repair_atom_embedding_metadata(
                    self.embedding_table_path,
                    self.embedding_metadata_path,
                    current_raw_corpus_sha256=self.raw_metadata[
                        "corpus_sha256"
                    ],
                )
            self._reload_state()
            self._validate_checkpoint_compatibility()
            self._require_fully_synchronized()

    def _validate_checkpoint_compatibility(self):
        if self.embedding_metadata.get("checkpoint_sha256") != (
            self.checkpoint_sha256
        ):
            raise ValueError(
                "标准原子词表与当前 checkpoint 哈希不匹配，"
                "请重新运行训练脚本"
            )
        if self.checkpoint["training_raw_corpus_sha256"] != (
            self.embedding_metadata.get(
                "training_raw_corpus_sha256"
            )
        ):
            raise ValueError(
                "checkpoint 与标准原子词表的训练语料哈希不匹配"
            )
        if self.checkpoint["model_config"] != (
            self.embedding_metadata.get("model_config")
        ):
            raise ValueError(
                "checkpoint 与标准原子词表的模型配置不匹配"
            )
        if sorted(self.observed_atomic_numbers) != sorted(
            int(value)
            for value in self.embedding_metadata.get(
                "observed_atomic_numbers", []
            )
        ):
            raise ValueError(
                "checkpoint 与标准原子词表的已见元素集合不匹配"
            )
        training_count = int(
            self.embedding_metadata["training_smiles_count"]
        )
        if (
            training_count < 1
            or training_count > len(self.embedding_table["smiles"])
            or training_count > len(self.raw_corpus["smiles"])
        ):
            raise ValueError("标准原子词表的训练片段计数非法")
        base_smiles_count = int(
            self.raw_metadata["base_smiles_count"]
        )
        if (
            base_smiles_count < 1
            or base_smiles_count > len(self.raw_corpus["smiles"])
        ):
            raise ValueError(
                "原子粗语料的基础片段计数非法: "
                f"{base_smiles_count}"
            )
        # 基础片段集合允许在 checkpoint 训练后扩展或重排。新增片段由冻结
        # 模型编码；下游按 SMILES 对齐并校验规范原子编号，checkpoint、哈希
        # 与事务校验仍负责拒绝重复、缺失、内容冲突或不完整写入。

    def _raw_smiles_missing_from_embedding_table(self):
        """按规范 SMILES 找出粗语料中尚无原子嵌入的片段。"""
        raw_smiles = self.raw_corpus["smiles"].astype(str).tolist()
        embedding_smiles = (
            self.embedding_table["smiles"].astype(str).tolist()
        )
        if len(self.raw_index) != len(raw_smiles):
            raise ValueError(
                "原子粗语料包含重复 SMILES，无法建立唯一索引"
            )
        if len(self.embedding_index) != len(embedding_smiles):
            raise ValueError(
                "标准原子词表包含重复 SMILES，无法建立唯一索引"
            )
        embedding_only = [
            smiles for smiles in embedding_smiles
            if smiles not in self.raw_index
        ]
        if embedding_only:
            raise ValueError(
                "标准原子词表存在原子粗语料中缺失的 SMILES，"
                f"数量={len(embedding_only)}，示例={embedding_only[:20]}"
            )
        return [
            smiles for smiles in raw_smiles
            if smiles not in self.embedding_index
        ]

    def _validate_canonical_atom_alignment(self):
        """按 SMILES 对齐两张表并验证对应的规范原子编号。"""
        missing_smiles = self._raw_smiles_missing_from_embedding_table()
        if missing_smiles:
            raise RuntimeError(
                f"标准原子词表仍缺少 {len(missing_smiles)} 个片段"
            )
        for canonical_smiles in self.raw_corpus[
            "smiles"
        ].astype(str).tolist():
            raw_index = self.raw_index[canonical_smiles]
            embedding_index = self.embedding_index[canonical_smiles]
            raw_start = int(self.raw_corpus["atom_offsets"][raw_index])
            raw_end = int(
                self.raw_corpus["atom_offsets"][raw_index + 1]
            )
            embedding_start = int(
                self.embedding_table["atom_offsets"][embedding_index]
            )
            embedding_end = int(
                self.embedding_table["atom_offsets"][embedding_index + 1]
            )
            if not np.array_equal(
                self.raw_corpus["canonical_atom_id"][raw_start:raw_end],
                self.embedding_table["canonical_atom_id"][
                    embedding_start:embedding_end
                ],
            ):
                raise ValueError(
                    f"片段 {canonical_smiles} 在两张表中的规范原子编号"
                    "不一致，数据已损坏"
                )

    def _require_fully_synchronized(self):
        self._validate_canonical_atom_alignment()
        if self.embedding_metadata.get(
            "current_raw_corpus_sha256"
        ) != self.raw_metadata.get("corpus_sha256"):
            raise ValueError(
                "原子粗语料与标准原子词表已按 SMILES 对齐，"
                "但标准表记录的当前粗语料哈希仍未同步"
            )

    def _read_transaction(self):
        with open(
            self.transaction_path, "r", encoding="utf-8"
        ) as file_obj:
            transaction = json.load(file_obj)
        if not isinstance(transaction, dict):
            raise ValueError("Resolver 事务记录格式非法")
        return transaction

    def _validate_transaction(self, transaction):
        if transaction.get("checkpoint_sha256") != (
            self.checkpoint_sha256
        ):
            raise ValueError(
                "未完成事务属于另一个 checkpoint，"
                "请人工检查数据文件后再继续"
            )
        canonical_smiles = transaction.get("canonical_smiles")
        if not isinstance(canonical_smiles, str) or not canonical_smiles:
            raise ValueError("Resolver 事务记录缺少规范 SMILES")

    def _write_transaction(self, canonical_smiles, stage):
        utils.atomic_write_json(
            self.transaction_path,
            {
                "canonical_smiles": canonical_smiles,
                "stage": str(stage),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "updated_at": time.time(),
                "checkpoint_sha256": self.checkpoint_sha256,
                "raw_corpus_sha256_before": self.raw_metadata[
                    "corpus_sha256"
                ],
                "embedding_table_sha256_before": (
                    self.embedding_metadata["table_sha256"]
                ),
            },
        )

    def _remove_transaction(self):
        try:
            os.remove(self.transaction_path)
        except FileNotFoundError:
            pass

    def _compatibility_anchor_profile(self):
        """描述当前权威原子 NPZ 的可验证语义状态。"""
        return {
            "archive_sha256": atom_train.file_sha256(
                self.embedding_table_path
            ),
            "num_smiles": int(
                self.embedding_metadata["num_smiles"]
            ),
            "num_atoms": int(self.embedding_metadata["num_atoms"]),
            "embedding_dim": int(
                self.embedding_metadata["embedding_dim"]
            ),
            "array_sha256": dict(
                self.embedding_metadata["array_sha256"]
            ),
            "table_sha256": str(
                self.embedding_metadata["table_sha256"]
            ),
            "atom_encoder_checkpoint_sha256": (
                self.checkpoint_sha256
            ),
        }

    def _read_compatibility_anchors(self):
        if not os.path.isfile(self.compatibility_anchors_path):
            return {
                "table_file_name": os.path.basename(
                    self.embedding_table_path
                ),
                "anchors": {},
            }
        with open(
            self.compatibility_anchors_path,
            "r",
            encoding="utf-8",
        ) as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise ValueError("原子词表兼容锚点文件格式非法")
        if payload.get("table_file_name") != os.path.basename(
            self.embedding_table_path
        ):
            raise ValueError("原子词表兼容锚点对应了另一张 NPZ")
        if not isinstance(payload.get("anchors"), dict):
            raise ValueError("原子词表兼容锚点记录格式非法")
        return payload

    def _record_compatibility_anchor_locked(self):
        """在持有原子表写锁时记录扩展前状态。"""
        profile = self._compatibility_anchor_profile()
        archive_sha256 = profile["archive_sha256"]
        payload = self._read_compatibility_anchors()
        existing = payload["anchors"].get(archive_sha256)
        if existing is not None:
            if existing != profile:
                raise ValueError(
                    "同一原子词表文件哈希对应了不同兼容锚点，"
                    "拒绝继续追加"
                )
            return profile
        payload["anchors"][archive_sha256] = profile
        utils.atomic_write_json(
            self.compatibility_anchors_path, payload
        )
        return profile

    def _repair_atomic_pairs_locked(self):
        try:
            raw_corpus, raw_metadata = (
                utils.load_raw_fragment_atom_corpus(
                    self.raw_corpus_path,
                    self.raw_metadata_path,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            raw_corpus, raw_metadata = (
                utils.repair_raw_fragment_atom_metadata(
                    self.raw_corpus_path,
                    self.raw_metadata_path,
                )
            )

        try:
            atom_train.load_atom_embedding_table(
                self.embedding_table_path,
                self.embedding_metadata_path,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            atom_train.repair_atom_embedding_metadata(
                self.embedding_table_path,
                self.embedding_metadata_path,
                current_raw_corpus_sha256=raw_metadata[
                    "corpus_sha256"
                ],
            )
        del raw_corpus

    def _warn_for_record(self, record):
        atomic_numbers = {
            int(value) for value in record["atom_features"][:, 0]
        }
        unseen_elements = sorted(
            atomic_numbers - self.observed_atomic_numbers
        )
        if unseen_elements:
            warnings.warn(
                "片段包含训练时未见元素 "
                f"{unseen_elements}；已使用预分配元素向量继续推理，"
                "结果可信度较低。",
                RuntimeWarning,
                stacklevel=3,
            )

        numeric_values = record["atom_features"][
            :, atom_train.ATOM_NUMERIC_COLUMNS
        ].astype(np.float64)
        numeric_mean = np.asarray(
            self.checkpoint["model_config"]["atom_numeric_mean"],
            dtype=np.float64,
        )
        numeric_std = np.asarray(
            self.checkpoint["model_config"]["atom_numeric_std"],
            dtype=np.float64,
        )
        z_scores = np.abs(
            (numeric_values - numeric_mean) / numeric_std
        )
        outlier_columns = np.flatnonzero(
            np.any(z_scores > atom_train.NUMERIC_CLIP_SIGMA, axis=0)
        )
        if len(outlier_columns) > 0:
            numeric_names = [
                utils.RAW_ATOM_FEATURE_COLUMNS[column_index]
                for column_index in atom_train.ATOM_NUMERIC_COLUMNS
            ]
            details = ", ".join(
                f"{numeric_names[column]}="
                f"{float(z_scores[:, column].max()):.2f}σ"
                for column in outlier_columns.tolist()
            )
            warnings.warn(
                "原子数值属性超出训练分布 "
                f"{atom_train.NUMERIC_CLIP_SIGMA:g}σ（{details}）；"
                "数值投影输入统一按 "
                f"±{atom_train.NUMERIC_CLIP_SIGMA:g}σ 裁剪。",
                RuntimeWarning,
                stacklevel=3,
            )

    def _infer_record(self, record):
        self._warn_for_record(record)
        data = atom_train.build_data_from_atom_record(record).to(
            self.device
        )
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(
                data.atom_features,
                data.bond_index,
                data.bond_features,
            )
        embeddings = (
            embeddings.detach().cpu().numpy().astype(np.float32)
        )
        expected_shape = (
            len(record["canonical_atom_id"]),
            atom_train.ATOM_EMBEDDING_DIM,
        )
        if embeddings.shape != expected_shape:
            raise RuntimeError(
                "checkpoint 输出的原子嵌入形状错误: "
                f"{embeddings.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise RuntimeError("checkpoint 输出包含 NaN 或 Inf")
        return embeddings

    def _recover_locked(self):
        transaction = None
        if os.path.isfile(self.transaction_path):
            transaction = self._read_transaction()
            self._validate_transaction(transaction)

        try:
            self._reload_state()
        except (OSError, ValueError, json.JSONDecodeError):
            if transaction is None:
                raise
            self._repair_atomic_pairs_locked()
            self._reload_state()
        self._validate_checkpoint_compatibility()

        missing_smiles = self._raw_smiles_missing_from_embedding_table()
        recovered_smiles = set()
        if missing_smiles:
            self._record_compatibility_anchor_locked()
        for canonical_smiles in missing_smiles:
            raw_index = self.raw_index[canonical_smiles]
            record = utils.extract_fragment_atom_record(
                self.raw_corpus, raw_index
            )
            embeddings = self._infer_record(record)
            atom_train.append_atom_embedding_record(
                canonical_smiles,
                record["canonical_atom_id"],
                embeddings,
                self.embedding_table_path,
                self.embedding_metadata_path,
                current_raw_corpus_sha256=self.raw_metadata[
                    "corpus_sha256"
                ],
            )
            recovered_smiles.add(canonical_smiles)

        self._reload_state()
        self._validate_checkpoint_compatibility()
        self._validate_canonical_atom_alignment()
        if self.embedding_metadata.get(
            "current_raw_corpus_sha256"
        ) != self.raw_metadata["corpus_sha256"]:
            if not missing_smiles:
                self._record_compatibility_anchor_locked()
            atom_train.repair_atom_embedding_metadata(
                self.embedding_table_path,
                self.embedding_metadata_path,
                current_raw_corpus_sha256=self.raw_metadata[
                    "corpus_sha256"
                ],
            )
            self._reload_state()
            self._validate_checkpoint_compatibility()

        self._require_fully_synchronized()
        if transaction is not None:
            self._remove_transaction()
        return recovered_smiles

    def _result_from_cache(self, canonical_smiles, source):
        if canonical_smiles not in self.raw_index:
            raise ValueError(
                "标准原子词表存在该 SMILES，但原子粗语料缺失，"
                "数据已损坏，拒绝返回"
            )
        embedding_index = self.embedding_index[canonical_smiles]
        atom_start = int(
            self.embedding_table["atom_offsets"][embedding_index]
        )
        atom_end = int(
            self.embedding_table["atom_offsets"][embedding_index + 1]
        )
        return {
            "canonical_smiles": canonical_smiles,
            "canonical_atom_id": np.array(
                self.embedding_table["canonical_atom_id"][
                    atom_start:atom_end
                ],
                dtype=np.int32,
                copy=True,
            ),
            "atom_embeddings": np.array(
                self.embedding_table["atom_embeddings"][
                    atom_start:atom_end
                ],
                dtype=np.float32,
                copy=True,
            ),
            "source": source,
        }

    @staticmethod
    def _validate_input_smiles(smiles, position=None):
        if not isinstance(smiles, str) or not smiles.strip():
            prefix = "" if position is None else f"第 {position} 个"
            raise ValueError(f"{prefix}SMILES 必须是非空字符串")
        smiles = smiles.strip()
        if smiles in utils.RAW_FRAGMENT_SPECIAL_TOKENS:
            raise ValueError(
                f"特殊标记不能作为真实片段查询或新增: {smiles}"
            )
        return smiles

    def resolve(self, smiles):
        return self.resolve_many([smiles])[0]

    def resolve_many(self, smiles_values):
        if isinstance(smiles_values, str):
            smiles_values = [smiles_values]
        else:
            smiles_values = list(smiles_values)
        if not smiles_values:
            self.last_resolution = {
                "requested_count": 0,
                "canonical_smiles": [],
                "source": [],
                "newly_added_to_standard_table": [],
                "newly_added_to_raw_corpus": [],
            }
            return []
        with self._thread_lock:
            return self._resolve_many_thread_locked(smiles_values)

    def _resolve_many_thread_locked(self, smiles_values):
        results = [None] * len(smiles_values)
        canonical_values = [None] * len(smiles_values)
        sources = [None] * len(smiles_values)
        records_by_smiles = {}

        for position, raw_smiles in enumerate(smiles_values):
            raw_smiles = self._validate_input_smiles(
                raw_smiles, position=position
            )
            record = utils.calculate_fragment_atom_record(raw_smiles)
            canonical_smiles = record["smiles"]
            canonical_values[position] = canonical_smiles
            if canonical_smiles in self.embedding_index:
                self._warn_for_record(record)
                results[position] = self._result_from_cache(
                    canonical_smiles, source="cache"
                )
                sources[position] = "cache"
            else:
                records_by_smiles.setdefault(canonical_smiles, record)

        unresolved = [
            smiles for smiles in records_by_smiles
            if smiles not in self.embedding_index
        ]
        inferred_by_smiles = {}
        for canonical_smiles in unresolved:
            inferred_by_smiles[canonical_smiles] = self._infer_record(
                records_by_smiles[canonical_smiles]
            )

        newly_added_to_table = []
        newly_added_to_raw = []
        recovered_smiles = set()
        if unresolved:
            with self._new_lock():
                recovered_smiles = self._recover_locked()
                still_unresolved = [
                    smiles for smiles in unresolved
                    if smiles not in self.embedding_index
                ]
                if still_unresolved:
                    self._record_compatibility_anchor_locked()

                for canonical_smiles in still_unresolved:
                    if canonical_smiles in self.raw_index:
                        raise RuntimeError(
                            "原子粗语料已有记录但标准词表恢复后仍缺失"
                        )
                    record = records_by_smiles[canonical_smiles]
                    self._write_transaction(
                        canonical_smiles, stage="prepared"
                    )
                    append_result, raw_metadata = (
                        utils.append_raw_fragment_atom_record(
                            canonical_smiles,
                            self.raw_corpus_path,
                            self.raw_metadata_path,
                        )
                    )
                    if not append_result["added"]:
                        raise RuntimeError(
                            "锁内二次查询后仍出现重复原子粗语料追加"
                        )
                    newly_added_to_raw.append(canonical_smiles)

                    self._reload_state()
                    self._validate_checkpoint_compatibility()
                    self._write_transaction(
                        canonical_smiles, stage="raw_appended"
                    )
                    atom_train.append_atom_embedding_record(
                        canonical_smiles,
                        record["canonical_atom_id"],
                        inferred_by_smiles[canonical_smiles],
                        self.embedding_table_path,
                        self.embedding_metadata_path,
                        current_raw_corpus_sha256=raw_metadata[
                            "corpus_sha256"
                        ],
                    )
                    newly_added_to_table.append(canonical_smiles)
                    self._write_transaction(
                        canonical_smiles, stage="embedding_appended"
                    )
                    self._remove_transaction()
                    self._reload_state()
                    self._validate_checkpoint_compatibility()

                if still_unresolved:
                    self._require_fully_synchronized()

                for position, canonical_smiles in enumerate(
                    canonical_values
                ):
                    if results[position] is not None:
                        continue
                    source = (
                        "checkpoint"
                        if canonical_smiles in recovered_smiles
                        or canonical_smiles in newly_added_to_table
                        else "cache"
                    )
                    results[position] = self._result_from_cache(
                        canonical_smiles, source=source
                    )
                    sources[position] = source

        if any(result is None for result in results):
            raise RuntimeError("原子 Resolver 未能填充全部请求位置")
        if any(source is None for source in sources):
            raise RuntimeError("原子 Resolver 未能标记全部结果来源")
        self.last_resolution = {
            "requested_count": len(smiles_values),
            "canonical_smiles": canonical_values,
            "source": sources,
            "newly_added_to_standard_table": newly_added_to_table,
            "newly_added_to_raw_corpus": newly_added_to_raw,
        }
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="查询或动态生成片段的标准64维原子词嵌入"
    )
    parser.add_argument("smiles", nargs="+")
    parser.add_argument("--raw-corpus", default=DEFAULT_RAW_CORPUS)
    parser.add_argument("--raw-metadata", default=DEFAULT_RAW_METADATA)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="原子编码器 PyTorch checkpoint（默认使用 .pth）",
    )
    parser.add_argument(
        "--embedding-table", default=DEFAULT_EMBEDDING_TABLE
    )
    parser.add_argument(
        "--embedding-metadata", default=DEFAULT_EMBEDDING_METADATA
    )
    parser.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    parser.add_argument(
        "--transaction-file", default=DEFAULT_TRANSACTION_FILE
    )
    parser.add_argument(
        "--compatibility-anchors",
        default=None,
        help="原子词表追加兼容锚点；默认与 embedding NPZ 同目录同前缀",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    resolver = AtomEmbeddingResolver(
        raw_corpus_path=args.raw_corpus,
        raw_metadata_path=args.raw_metadata,
        checkpoint_path=args.checkpoint,
        embedding_table_path=args.embedding_table,
        embedding_metadata_path=args.embedding_metadata,
        lock_path=args.lock_file,
        transaction_path=args.transaction_file,
        compatibility_anchors_path=args.compatibility_anchors,
        device=args.device,
    )
    results = resolver.resolve_many(args.smiles)
    for smiles, result in zip(args.smiles, results):
        print(
            f"{smiles} -> {result['canonical_smiles']} | "
            f"atoms={len(result['canonical_atom_id'])} | "
            f"shape={tuple(result['atom_embeddings'].shape)} | "
            f"source={result['source']}"
        )


if __name__ == "__main__":
    main()
