"""
查询或动态补充标准128维片段词嵌入。

缓存命中时直接读取权威标准 NPZ；缺失词汇由冻结 checkpoint 编码，
并在跨进程锁内向片段粗语料和标准词嵌入表尾部进行事务式追加。每次
标准表更新后，同时完整重建一片段一行的可读 JSON 镜像。
"""

import argparse
import hashlib
import json
import os
import socket
import threading
import time
import uuid
import warnings

import numpy as np

try:
    import utils
    import PhiSSeparator_fragment_encoder_train as fragment_train
except ImportError:  # 支持作为 PhiSSeparator 子模块导入
    from . import utils
    from . import PhiSSeparator_fragment_encoder_train as fragment_train


EMBEDDING_DIM = fragment_train.OUTPUT_DIM
ENCODE_BATCH_SIZE = fragment_train.ENCODE_BATCH_SIZE
DEVICE = "auto"

CHECKPOINT_PATH = fragment_train.CHECKPOINT_PATH
RAW_CORPUS_PATH = fragment_train.RAW_CORPUS_PATH
RAW_CORPUS_METADATA_PATH = fragment_train.RAW_CORPUS_METADATA_PATH
EMBEDDING_TABLE_PATH = fragment_train.EMBEDDING_TABLE_PATH
EMBEDDING_JSON_PATH = fragment_train.EMBEDDING_JSON_PATH
EMBEDDING_METADATA_PATH = fragment_train.EMBEDDING_METADATA_PATH

LOCK_PATH = os.path.join(
    fragment_train.SHARED_DATA_DIR,
    f".fragment_embeddings_{EMBEDDING_DIM}d.lock",
)
TRANSACTION_PATH = os.path.join(
    fragment_train.SHARED_DATA_DIR,
    "fragment_embedding_update_transaction.json",
)
EXTENSION_LOG_PATH = os.path.join(
    fragment_train.SHARED_DATA_DIR,
    "fragment_embedding_extension_log.jsonl",
)

LOCK_TIMEOUT_SECONDS = 120.0
LOCK_STALE_SECONDS = 10 * 60.0

GLOBAL_NODE_TOKEN = "<GLOBAL>"
UNKNOWN_TOKEN = "<UNK>"
RAW_CORPUS_FIELDS = (
    "smiles",
    "ecfp",
    "fcfp",
    "hac",
    "ring_count",
    "formal_charge",
    "element_counts",
    "is_metal",
)


def _raw_corpus_checksum(corpus):
    checksum_function = getattr(
        utils, "_raw_fragment_corpus_checksum", None
    )
    if not callable(checksum_function):
        raise RuntimeError("utils.py 缺少片段粗语料校验函数")
    return checksum_function(corpus)


def _raw_array_sha256(values, is_string=False):
    digest = hashlib.sha256()
    values = np.asarray(values)
    if is_string:
        for value in values.astype(str).tolist():
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _record_from_corpus(corpus, row_index):
    return {
        "smiles": str(corpus["smiles"][row_index]),
        "ecfp": np.array(corpus["ecfp"][row_index], copy=True),
        "fcfp": np.array(corpus["fcfp"][row_index], copy=True),
        "hac": np.array(corpus["hac"][row_index], copy=True),
        "ring_count": np.array(
            corpus["ring_count"][row_index], copy=True
        ),
        "formal_charge": np.array(
            corpus["formal_charge"][row_index], copy=True
        ),
        "element_counts": np.array(
            corpus["element_counts"][row_index], copy=True
        ),
        "is_metal": np.array(
            corpus["is_metal"][row_index], copy=True
        ),
    }


def _build_extended_raw_metadata(
    previous_metadata,
    corpus,
    *,
    training_smiles_count,
    checkpoint_sha256,
):
    training_smiles_count = int(training_smiles_count)
    num_smiles = len(corpus["smiles"])
    if not 1 <= training_smiles_count <= num_smiles:
        raise ValueError("片段粗语料的训练片段计数非法")
    metadata = dict(previous_metadata)
    metadata.update({
        "num_smiles": int(num_smiles),
        "corpus_sha256": _raw_corpus_checksum(corpus),
        "array_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in corpus.items()
        },
        "array_dtypes": {
            key: np.asarray(value).dtype.str
            for key, value in corpus.items()
        },
        "array_sha256": {
            key: _raw_array_sha256(
                value, is_string=(key == "smiles")
            )
            for key, value in corpus.items()
        },
        "fragment_embedding_training_smiles_count": (
            training_smiles_count
        ),
        "fragment_embedding_resolver_added_smiles_count": (
            num_smiles - training_smiles_count
        ),
        "fragment_embedding_checkpoint_sha256": str(
            checkpoint_sha256
        ),
        "last_fragment_embedding_extension_utc": (
            fragment_train.utc_now()
        ),
    })
    return metadata


def _write_raw_corpus_atomic(
    corpus,
    metadata,
    npz_path,
    metadata_path,
):
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_npz = f"{npz_path}.{token}.tmp.npz"
    temp_metadata = f"{metadata_path}.{token}.tmp"
    try:
        np.savez_compressed(temp_npz, **corpus)
        with open(temp_metadata, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        utils.load_raw_fragment_corpus(temp_npz, temp_metadata)
        os.replace(temp_npz, npz_path)
        os.replace(temp_metadata, metadata_path)
    finally:
        for temp_path in (temp_npz, temp_metadata):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def _repair_raw_corpus_pair(
    npz_path,
    metadata_path,
    *,
    training_smiles_count,
    checkpoint_sha256,
):
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        previous_metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(RAW_CORPUS_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"待恢复片段粗语料缺少字段: {sorted(missing)}"
            )
        corpus = {
            key: np.array(loaded[key], copy=True)
            for key in RAW_CORPUS_FIELDS
        }
    corpus["smiles"] = corpus["smiles"].astype(str)
    metadata = _build_extended_raw_metadata(
        previous_metadata,
        corpus,
        training_smiles_count=training_smiles_count,
        checkpoint_sha256=checkpoint_sha256,
    )
    _write_raw_corpus_atomic(
        corpus, metadata, npz_path, metadata_path
    )
    return corpus, metadata


def _append_raw_fragment_records(
    records,
    *,
    npz_path,
    metadata_path,
    training_smiles_count,
    checkpoint_sha256,
):
    corpus, previous_metadata = utils.load_raw_fragment_corpus(
        npz_path, metadata_path
    )
    existing = set(corpus["smiles"].astype(str).tolist())
    pending_by_smiles = {}
    for record in records:
        canonical_smiles = str(record["smiles"])
        if canonical_smiles in existing:
            continue
        previous = pending_by_smiles.get(canonical_smiles)
        if previous is not None:
            for field in RAW_CORPUS_FIELDS[1:]:
                if not np.array_equal(
                    previous[field], record[field]
                ):
                    raise ValueError(
                        f"新增片段 {canonical_smiles} 的粗特征不一致"
                    )
            continue
        pending_by_smiles[canonical_smiles] = record

    if not pending_by_smiles:
        return [], previous_metadata
    pending = list(pending_by_smiles.values())
    new_arrays = fragment_train.records_to_corpus(pending)
    updated = {
        key: np.concatenate(
            [corpus[key], new_arrays[key]], axis=0
        )
        for key in RAW_CORPUS_FIELDS
    }
    metadata = _build_extended_raw_metadata(
        previous_metadata,
        updated,
        training_smiles_count=training_smiles_count,
        checkpoint_sha256=checkpoint_sha256,
    )
    _write_raw_corpus_atomic(
        updated, metadata, npz_path, metadata_path
    )
    return list(pending_by_smiles), metadata


def _append_extension_log(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ))
            file_obj.write("\n")
    except OSError as exc:
        warnings.warn(
            f"片段词嵌入已更新，但扩展日志写入失败: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


class FragmentEmbeddingResolver:
    """缓存优先、可批量解析且可并发扩展的片段词嵌入解析器。"""

    def __init__(
        self,
        *,
        checkpoint_path=CHECKPOINT_PATH,
        raw_corpus_path=RAW_CORPUS_PATH,
        raw_corpus_metadata_path=RAW_CORPUS_METADATA_PATH,
        embedding_table_path=EMBEDDING_TABLE_PATH,
        embedding_metadata_path=EMBEDDING_METADATA_PATH,
        extension_log_path=EXTENSION_LOG_PATH,
        lock_path=LOCK_PATH,
        transaction_path=TRANSACTION_PATH,
        embedding_dim=EMBEDDING_DIM,
        batch_size=ENCODE_BATCH_SIZE,
        device_spec=DEVICE,
        lock_timeout_seconds=LOCK_TIMEOUT_SECONDS,
        stale_lock_seconds=LOCK_STALE_SECONDS,
    ):
        self.checkpoint_path = os.path.abspath(
            os.fspath(checkpoint_path)
        )
        self.raw_corpus_path = os.path.abspath(
            os.fspath(raw_corpus_path)
        )
        self.raw_corpus_metadata_path = os.path.abspath(
            os.fspath(raw_corpus_metadata_path)
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
        self.extension_log_path = os.path.abspath(
            os.fspath(extension_log_path)
        )
        self.lock_path = os.path.abspath(os.fspath(lock_path))
        self.transaction_path = os.path.abspath(
            os.fspath(transaction_path)
        )
        self.embedding_dim = int(embedding_dim)
        self.batch_size = int(batch_size)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.device = fragment_train.select_device(device_spec)

        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"当前解析器只支持 {EMBEDDING_DIM} 维词嵌入"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")

        required_paths = (
            self.checkpoint_path,
            self.raw_corpus_path,
            self.raw_corpus_metadata_path,
            self.embedding_table_path,
            self.embedding_metadata_path,
        )
        missing = [
            path for path in required_paths
            if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "片段编码器产物不完整，请先运行 "
                "PhiSSeparator_fragment_encoder_train.py："
                f"{missing}"
            )

        self.checkpoint = fragment_train.load_checkpoint_payload(
            self.checkpoint_path,
            expected_dim=self.embedding_dim,
        )
        self.checkpoint_sha256 = fragment_train.file_sha256(
            self.checkpoint_path
        )
        self._model = None
        self._thread_lock = threading.RLock()
        self.last_resolution = None

        try:
            self._reload_state()
        except (OSError, ValueError, json.JSONDecodeError):
            if not os.path.isfile(self.transaction_path):
                raise
            with self._new_process_lock():
                transaction = self._read_transaction()
                self._validate_transaction(transaction)
                self._repair_pairs_locked()
                self._reload_state()
                self._validate_compatibility()
                self._recover_locked()
                self._reload_state()

        self._validate_compatibility()
        raw_tail = self._raw_tail_missing_from_standard_table()
        if os.path.isfile(self.transaction_path) or raw_tail:
            with self._new_process_lock():
                self._recover_locked()
                self._reload_state()
                self._validate_compatibility()
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
            utils.load_raw_fragment_corpus,
            self.raw_corpus_path,
            self.raw_corpus_metadata_path,
        )
        (
            self.embedding_table,
            self.embedding_metadata,
        ) = self._load_with_retry(
            fragment_train.load_fragment_embedding_table,
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

    def _new_process_lock(self):
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
            table_kind="fragment",
        ):
            return
        with self._new_process_lock():
            self._reload_state()
            if utils.embedding_json_mirror_needs_refresh(
                self.embedding_json_path,
                self.embedding_table,
                self.embedding_metadata,
                table_kind="fragment",
            ):
                fragment_train.repair_fragment_embedding_metadata(
                    self.embedding_table_path,
                    self.embedding_metadata_path,
                    checkpoint=self.checkpoint,
                    current_raw_corpus_sha256=self.raw_metadata[
                        "corpus_sha256"
                    ],
                )
            self._reload_state()
            self._validate_compatibility()
            self._require_fully_synchronized()

    def _validate_compatibility(self):
        if self.embedding_metadata.get("checkpoint_sha256") != (
            self.checkpoint_sha256
        ):
            raise ValueError(
                "标准片段词嵌入表与 checkpoint 哈希不匹配，"
                "请重新运行训练脚本"
            )
        if self.embedding_metadata.get(
            "training_raw_corpus_sha256"
        ) != self.checkpoint["training_corpus_sha256"]:
            raise ValueError(
                "标准片段词嵌入表与 checkpoint 的训练语料不匹配"
            )
        if self.embedding_metadata.get("model_config") != (
            self.checkpoint["model_config"]
        ):
            raise ValueError("标准表与 checkpoint 的模型结构不匹配")
        if self.embedding_metadata.get("normalization") != (
            self.checkpoint["normalization"]
        ):
            raise ValueError("标准表与 checkpoint 的归一化参数不匹配")
        if self.embedding_metadata.get("feature_config") != (
            self.checkpoint["feature_config"]
        ):
            raise ValueError("标准表与 checkpoint 的粗特征配置不匹配")
        fragment_train.validate_checkpoint_against_corpus_metadata(
            self.checkpoint, self.raw_metadata
        )
        training_count = int(
            self.embedding_metadata["training_smiles_count"]
        )
        if (
            training_count < 1
            or training_count > len(self.embedding_table["smiles"])
            or training_count > len(self.raw_corpus["smiles"])
        ):
            raise ValueError("标准表训练片段计数非法")
        if self.raw_metadata.get(
            "fragment_embedding_checkpoint_sha256"
        ) == self.checkpoint_sha256:
            if self.raw_metadata.get(
                "fragment_embedding_training_smiles_count"
            ) != training_count:
                raise ValueError(
                    "片段粗语料的 Resolver 训练基线计数不一致"
                )
            if self.raw_metadata.get(
                "fragment_embedding_resolver_added_smiles_count"
            ) != len(self.raw_corpus["smiles"]) - training_count:
                raise ValueError(
                    "片段粗语料的 Resolver 新增计数不一致"
                )

    def _raw_tail_missing_from_standard_table(self):
        raw_smiles = self.raw_corpus["smiles"].astype(str).tolist()
        table_smiles = (
            self.embedding_table["smiles"].astype(str).tolist()
        )
        common_count = min(len(raw_smiles), len(table_smiles))
        if raw_smiles[:common_count] != table_smiles[:common_count]:
            raise ValueError(
                "片段粗语料与标准词嵌入表的顺序或内容不一致，"
                "请重新运行训练脚本"
            )
        if len(table_smiles) > len(raw_smiles):
            raise ValueError(
                "标准词嵌入表存在粗语料中缺失的 SMILES，"
                "数据已损坏，拒绝返回"
            )
        return raw_smiles[len(table_smiles):]

    def _tail_is_resolver_managed(self):
        return (
            self.raw_metadata.get(
                "fragment_embedding_checkpoint_sha256"
            )
            == self.checkpoint_sha256
            and self.raw_metadata.get(
                "fragment_embedding_training_smiles_count"
            )
            == self.embedding_metadata["training_smiles_count"]
        )

    def _require_fully_synchronized(self):
        missing_tail = self._raw_tail_missing_from_standard_table()
        if missing_tail:
            raise RuntimeError(
                f"标准片段词嵌入表仍缺少 {len(missing_tail)} 个片段"
            )
        if self.embedding_metadata.get(
            "current_raw_corpus_sha256"
        ) != self.raw_metadata.get("corpus_sha256"):
            raise ValueError(
                "片段粗语料已发生非 Resolver 追加式变化，"
                "必须重新运行训练脚本"
            )

    def _ensure_model(self):
        if self._model is None:
            self._model = fragment_train.build_model_from_checkpoint(
                self.checkpoint, self.device
            )
        return self._model

    def _warn_for_records(self, records):
        normalization = self.checkpoint["normalization"]
        means = np.asarray(
            normalization["mean"], dtype=np.float64
        )
        stds = np.asarray(
            normalization["std"], dtype=np.float64
        )
        for record in records:
            values = np.asarray([
                float(record[name])
                for name in fragment_train.SCALAR_FEATURE_ORDER
            ])
            z_scores = np.abs((values - means) / stds)
            outliers = np.flatnonzero(
                z_scores > fragment_train.NUMERIC_CLIP_SIGMA
            )
            if len(outliers) > 0:
                details = ", ".join(
                    f"{fragment_train.SCALAR_FEATURE_ORDER[index]}="
                    f"{z_scores[index]:.2f}σ"
                    for index in outliers.tolist()
                )
                warnings.warn(
                    f"片段 {record['smiles']} 的数值粗特征超出训练"
                    f"分布 {fragment_train.NUMERIC_CLIP_SIGMA:g}σ"
                    f"（{details}）；编码输入将按 "
                    f"±{fragment_train.NUMERIC_CLIP_SIGMA:g}σ 裁剪。",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def _infer_records(self, records):
        if not records:
            return np.empty(
                (0, self.embedding_dim), dtype=np.float32
            )
        self._warn_for_records(records)
        return fragment_train.encode_feature_records(
            records,
            self._ensure_model(),
            self.checkpoint["normalization"],
            self.device,
            batch_size=self.batch_size,
        )

    def _read_transaction(self):
        with open(
            self.transaction_path, "r", encoding="utf-8"
        ) as file_obj:
            transaction = json.load(file_obj)
        if not isinstance(transaction, dict):
            raise ValueError("片段 Resolver 事务记录格式非法")
        return transaction

    def _validate_transaction(self, transaction):
        if transaction.get("checkpoint_sha256") != (
            self.checkpoint_sha256
        ):
            raise ValueError(
                "未完成事务属于另一个 checkpoint，拒绝自动恢复"
            )
        smiles_values = transaction.get("canonical_smiles")
        if (
            not isinstance(smiles_values, list)
            or not smiles_values
            or any(
                not isinstance(value, str) or not value
                for value in smiles_values
            )
        ):
            raise ValueError("片段 Resolver 事务缺少规范 SMILES")

    def _write_transaction(self, canonical_smiles, stage):
        utils.atomic_write_json(
            self.transaction_path,
            {
                "canonical_smiles": list(canonical_smiles),
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

    def _repair_pairs_locked(self):
        try:
            with open(
                self.embedding_metadata_path,
                "r",
                encoding="utf-8",
            ) as file_obj:
                old_embedding_metadata = json.load(file_obj)
            training_count = int(
                old_embedding_metadata["training_smiles_count"]
            )
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "无法从旧标准表元数据确定训练片段计数"
            ) from exc

        try:
            raw_corpus, raw_metadata = (
                utils.load_raw_fragment_corpus(
                    self.raw_corpus_path,
                    self.raw_corpus_metadata_path,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            raw_corpus, raw_metadata = _repair_raw_corpus_pair(
                self.raw_corpus_path,
                self.raw_corpus_metadata_path,
                training_smiles_count=training_count,
                checkpoint_sha256=self.checkpoint_sha256,
            )

        try:
            fragment_train.load_fragment_embedding_table(
                self.embedding_table_path,
                self.embedding_metadata_path,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            fragment_train.repair_fragment_embedding_metadata(
                self.embedding_table_path,
                self.embedding_metadata_path,
                checkpoint=self.checkpoint,
                current_raw_corpus_sha256=raw_metadata[
                    "corpus_sha256"
                ],
            )
        del raw_corpus

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
            self._repair_pairs_locked()
            self._reload_state()
        self._validate_compatibility()

        missing_tail = self._raw_tail_missing_from_standard_table()
        if (
            missing_tail
            and transaction is None
            and not self._tail_is_resolver_managed()
        ):
            raise ValueError(
                "粗语料出现未经当前 Resolver 标记的新增片段，"
                "请重新运行训练脚本"
            )

        recovered = set()
        if missing_tail:
            records = [
                _record_from_corpus(
                    self.raw_corpus, self.raw_index[smiles]
                )
                for smiles in missing_tail
            ]
            embeddings = self._infer_records(records)
            entries = [
                {
                    "smiles": record["smiles"],
                    "fragment_embedding": embedding,
                }
                for record, embedding in zip(records, embeddings)
            ]
            added, _ = (
                fragment_train.append_fragment_embedding_records(
                    entries,
                    npz_path=self.embedding_table_path,
                    metadata_path=self.embedding_metadata_path,
                    checkpoint=self.checkpoint,
                    current_raw_corpus_sha256=self.raw_metadata[
                        "corpus_sha256"
                    ],
                )
            )
            recovered.update(added)

        self._reload_state()
        self._validate_compatibility()
        if (
            transaction is not None
            and not missing_tail
            and self.embedding_metadata.get(
                "current_raw_corpus_sha256"
            ) != self.raw_metadata["corpus_sha256"]
        ):
            fragment_train.repair_fragment_embedding_metadata(
                self.embedding_table_path,
                self.embedding_metadata_path,
                checkpoint=self.checkpoint,
                current_raw_corpus_sha256=self.raw_metadata[
                    "corpus_sha256"
                ],
            )
            self._reload_state()
            self._validate_compatibility()

        self._require_fully_synchronized()
        if transaction is not None:
            self._remove_transaction()
        return recovered

    def _embedding_copy(self, canonical_smiles):
        if canonical_smiles not in self.raw_index:
            raise ValueError(
                "标准表存在该 SMILES，但片段粗语料缺失，"
                "数据已损坏"
            )
        index = self.embedding_index[canonical_smiles]
        return np.array(
            self.embedding_table["fragment_embeddings"][index],
            dtype=np.float32,
            copy=True,
        )

    @staticmethod
    def _validate_input_smiles(smiles, position=None):
        if not isinstance(smiles, str) or not smiles.strip():
            prefix = (
                "" if position is None else f"第 {position} 个"
            )
            raise ValueError(f"{prefix}SMILES 必须是非空字符串")
        smiles = smiles.strip()
        if smiles in utils.RAW_FRAGMENT_SPECIAL_TOKENS:
            raise ValueError(
                f"特殊标记不能作为真实片段查询或新增: {smiles}"
            )
        return smiles

    def resolve(self, smiles):
        vector = self.resolve_one(smiles)
        return {
            "canonical_smiles": self.last_resolution[
                "canonical_smiles"
            ][0],
            "fragment_embedding": vector,
            "source": self.last_resolution["source"][0],
        }

    def resolve_one(self, smiles):
        return self.resolve_many([smiles])[0].copy()

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
                "newly_added_to_vocabulary": [],
                "newly_added_to_raw_corpus": [],
            }
            return np.empty(
                (0, self.embedding_dim), dtype=np.float32
            )
        with self._thread_lock:
            return self._resolve_many_thread_locked(smiles_values)

    def _resolve_many_thread_locked(self, smiles_values):
        vectors = [None] * len(smiles_values)
        canonical_values = [None] * len(smiles_values)
        sources = [None] * len(smiles_values)
        records_by_smiles = {}

        for position, raw_smiles in enumerate(smiles_values):
            raw_smiles = self._validate_input_smiles(
                raw_smiles, position=position
            )
            if raw_smiles in self.embedding_index:
                canonical_values[position] = raw_smiles
                vectors[position] = self._embedding_copy(raw_smiles)
                sources[position] = "cache"
                continue

            record = utils.calculate_fragment_vocab_features(
                raw_smiles
            )
            canonical_smiles = record["smiles"]
            canonical_values[position] = canonical_smiles
            if canonical_smiles in self.embedding_index:
                vectors[position] = self._embedding_copy(
                    canonical_smiles
                )
                sources[position] = "cache"
            else:
                records_by_smiles.setdefault(
                    canonical_smiles, record
                )

        unresolved = [
            smiles for smiles in records_by_smiles
            if smiles not in self.embedding_index
        ]
        newly_added_to_table = []
        newly_added_to_raw = []
        recovered = set()

        if unresolved:
            records = [
                records_by_smiles[smiles] for smiles in unresolved
            ]
            embeddings = self._infer_records(records)
            inferred_by_smiles = {
                record["smiles"]: embedding
                for record, embedding in zip(records, embeddings)
            }

            with self._new_process_lock():
                recovered = self._recover_locked()
                still_unresolved = [
                    smiles for smiles in unresolved
                    if smiles not in self.embedding_index
                ]
                if still_unresolved:
                    self._write_transaction(
                        still_unresolved, stage="prepared"
                    )
                    newly_added_to_raw, raw_metadata = (
                        _append_raw_fragment_records(
                            [
                                records_by_smiles[smiles]
                                for smiles in still_unresolved
                            ],
                            npz_path=self.raw_corpus_path,
                            metadata_path=(
                                self.raw_corpus_metadata_path
                            ),
                            training_smiles_count=(
                                self.embedding_metadata[
                                    "training_smiles_count"
                                ]
                            ),
                            checkpoint_sha256=(
                                self.checkpoint_sha256
                            ),
                        )
                    )
                    self._reload_state()
                    self._validate_compatibility()
                    self._write_transaction(
                        still_unresolved, stage="raw_appended"
                    )
                    entries = [
                        {
                            "smiles": smiles,
                            "fragment_embedding": (
                                inferred_by_smiles[smiles]
                            ),
                        }
                        for smiles in still_unresolved
                    ]
                    newly_added_to_table, _ = (
                        fragment_train.append_fragment_embedding_records(
                            entries,
                            npz_path=self.embedding_table_path,
                            metadata_path=self.embedding_metadata_path,
                            checkpoint=self.checkpoint,
                            current_raw_corpus_sha256=raw_metadata[
                                "corpus_sha256"
                            ],
                        )
                    )
                    self._write_transaction(
                        still_unresolved,
                        stage="embedding_appended",
                    )
                    self._remove_transaction()
                    self._reload_state()
                    self._validate_compatibility()
                    self._require_fully_synchronized()
                    _append_extension_log(
                        self.extension_log_path,
                        {
                            "timestamp_utc": fragment_train.utc_now(),
                            "checkpoint_sha256": (
                                self.checkpoint_sha256
                            ),
                            "added_to_standard_table": (
                                newly_added_to_table
                            ),
                            "added_to_raw_corpus": (
                                newly_added_to_raw
                            ),
                            "current_raw_corpus_sha256": (
                                self.raw_metadata["corpus_sha256"]
                            ),
                        },
                    )

                for position, canonical_smiles in enumerate(
                    canonical_values
                ):
                    if vectors[position] is not None:
                        continue
                    vectors[position] = self._embedding_copy(
                        canonical_smiles
                    )
                    sources[position] = (
                        "checkpoint"
                        if canonical_smiles in recovered
                        or canonical_smiles in newly_added_to_table
                        else "cache"
                    )

        self.last_resolution = {
            "requested_count": len(smiles_values),
            "canonical_smiles": canonical_values,
            "source": sources,
            "newly_added_to_standard_table": (
                newly_added_to_table
            ),
            # 保留旧字段名，便于现有调用方平滑迁移。
            "newly_added_to_vocabulary": newly_added_to_table,
            "newly_added_to_raw_corpus": newly_added_to_raw,
        }
        if any(vector is None for vector in vectors):
            raise RuntimeError("片段 Resolver 未能填充全部请求位置")
        if any(source is None for source in sources):
            raise RuntimeError("片段 Resolver 未能标记全部结果来源")
        return np.stack(vectors).astype(np.float32, copy=False)


_DEFAULT_RESOLVER = None
_DEFAULT_RESOLVER_LOCK = threading.Lock()


def get_default_resolver():
    global _DEFAULT_RESOLVER
    with _DEFAULT_RESOLVER_LOCK:
        if _DEFAULT_RESOLVER is None:
            _DEFAULT_RESOLVER = FragmentEmbeddingResolver()
        return _DEFAULT_RESOLVER


def get_fragment_embedding(smiles):
    return get_default_resolver().resolve_one(smiles)


def get_fragment_embeddings(smiles_values):
    return get_default_resolver().resolve_many(smiles_values)


def load_standard_fragment_embeddings(
    npz_path=EMBEDDING_TABLE_PATH,
    metadata_path=EMBEDDING_METADATA_PATH,
):
    """兼容下游批处理：返回 ``{canonical_smiles: vector}``。"""
    arrays, _ = fragment_train.load_fragment_embedding_table(
        npz_path, metadata_path
    )
    return {
        str(smiles): np.array(vector, copy=True)
        for smiles, vector in zip(
            arrays["smiles"], arrays["fragment_embeddings"]
        )
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="查询或动态新增标准128维片段词嵌入"
    )
    parser.add_argument("smiles", nargs="+")
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    args = parse_args()
    resolver = FragmentEmbeddingResolver(device_spec=args.device)
    embeddings = resolver.resolve_many(args.smiles)
    output = []
    for raw_smiles, canonical_smiles, source, vector in zip(
        args.smiles,
        resolver.last_resolution["canonical_smiles"],
        resolver.last_resolution["source"],
        embeddings,
    ):
        output.append({
            "input_smiles": raw_smiles,
            "canonical_smiles": canonical_smiles,
            "source": source,
            "embedding_dim": int(len(vector)),
            "fragment_embedding": vector.tolist(),
        })
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
