"""Train the local ranker or globally constrained PhiSLinker model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_PATH_CONFIG,
        DEFAULT_TRAIN_CONFIG,
        ModelConfig,
        PathConfig,
        TRAINING_STAGE,
        TrainConfig,
        configuration_snapshot,
    )
    from PhiSLinker_atom_pair_dataset import (
        AtomPairGraphDataset,
        ConnectionFeatureBuilder,
        build_dataloader,
        create_or_load_split,
        discover_pyg_files,
        run_preflight,
    )
    from PhiSLinker_atom_pair_engine import (
        TRAINING_STAGES,
        train_one_epoch,
        validate_one_epoch,
        validate_one_epoch_detailed,
    )
    from PhiSLinker_atom_pair_model import PhiSLinkerAtomPairModel
    from PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        atomic_torch_save,
        atomic_write_json,
        file_sha256,
        seed_everything,
        torch_load_compat,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_PATH_CONFIG,
        DEFAULT_TRAIN_CONFIG,
        ModelConfig,
        PathConfig,
        TRAINING_STAGE,
        TrainConfig,
        configuration_snapshot,
    )
    from .PhiSLinker_atom_pair_dataset import (
        AtomPairGraphDataset,
        ConnectionFeatureBuilder,
        build_dataloader,
        create_or_load_split,
        discover_pyg_files,
        run_preflight,
    )
    from .PhiSLinker_atom_pair_engine import (
        TRAINING_STAGES,
        train_one_epoch,
        validate_one_epoch,
        validate_one_epoch_detailed,
    )
    from .PhiSLinker_atom_pair_model import PhiSLinkerAtomPairModel
    from .PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        atomic_torch_save,
        atomic_write_json,
        file_sha256,
        seed_everything,
        torch_load_compat,
    )

def _parse_args() -> argparse.Namespace:
    default_training_stage = str(TRAINING_STAGE).strip().lower()
    if default_training_stage not in TRAINING_STAGES:
        allowed = ", ".join(sorted(TRAINING_STAGES))
        raise ValueError(
            "TRAINING_STAGE in PhiSLinker_atom_pair_config.py must be one of "
            f"{allowed}; got {TRAINING_STAGE!r}"
        )
    parser = argparse.ArgumentParser(
        description="Train/evaluate local or global atom-pair ranking"
    )
    parser.add_argument(
        "--training-stage",
        choices=sorted(TRAINING_STAGES),
        default=default_training_stage,
        help=(
            "training stage; defaults to TRAINING_STAGE in "
            "PhiSLinker_atom_pair_config.py"
        ),
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--fragment-vocab", type=Path, default=None)
    parser.add_argument("--atom-vocab", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="disable automatic resume for the selected run",
    )
    args = parser.parse_args()
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    if args.eval_only and args.resume is not None:
        parser.error("--eval-only and --resume are mutually exclusive")
    if args.checkpoint is not None and not args.eval_only:
        parser.error("--checkpoint is only used with --eval-only")
    if args.resume is not None and args.rebuild_split:
        parser.error("a resumed run cannot rebuild its data split")
    if args.fresh_start and args.resume is not None:
        parser.error("--fresh-start and --resume are mutually exclusive")
    if args.fresh_start and args.eval_only:
        parser.error("--fresh-start and --eval-only are mutually exclusive")
    if args.init_checkpoint is not None and args.training_stage != "global":
        parser.error("--init-checkpoint is only valid for --training-stage global")
    if args.init_checkpoint is not None and (
        args.resume is not None or args.eval_only
    ):
        parser.error("--init-checkpoint cannot be combined with resume/eval-only")
    return args


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    payload = torch_load_compat(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    if (
        str(payload.get("atom_pair_supervision_policy", ""))
        != ATOM_PAIR_SUPERVISION_POLICY
    ):
        raise ValueError("checkpoint uses an incompatible supervision policy")
    if "model_state_dict" not in payload:
        raise KeyError("checkpoint lacks model_state_dict")
    return payload


def _checkpoint_configs(payload: Mapping[str, Any]) -> tuple[ModelConfig, TrainConfig]:
    try:
        snapshot = payload["configuration"]
        model_config = ModelConfig(**dict(snapshot["model"]))
        train_config = TrainConfig(**dict(snapshot["training"]))
    except Exception as exc:
        raise ValueError("checkpoint contains an invalid configuration snapshot") from exc
    model_config.validate()
    train_config.validate()
    return model_config, train_config


def _checkpoint_stage(payload: Mapping[str, Any]) -> str:
    stage = str(payload.get("training_stage", "")).strip().lower()
    if stage not in TRAINING_STAGES:
        raise ValueError("checkpoint lacks a valid training_stage")
    return stage


def _validated_run_name(value: str) -> str:
    result = value.strip()
    if not result or result in {".", ".."} or any(char in result for char in "/\\"):
        raise ValueError("run name must be one non-empty path component")
    return result


def _resolved_output_root(args: argparse.Namespace) -> Path:
    return (args.output_root or DEFAULT_PATH_CONFIG.output_root).expanduser().resolve()


def _find_automatic_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if (
        args.eval_only
        or args.resume is not None
        or args.fresh_start
        or args.rebuild_split
        or args.init_checkpoint is not None
    ):
        return None
    checkpoint_root = _resolved_output_root(args) / "checkpoints"
    if args.run_name is not None:
        candidate = (
            checkpoint_root
            / _validated_run_name(args.run_name)
            / "last_model.pt"
        )
        return candidate.resolve() if candidate.is_file() else None
    if not checkpoint_root.is_dir():
        return None
    compatible: list[Path] = []
    for path in checkpoint_root.glob("*/last_model.pt"):
        if not path.is_file():
            continue
        try:
            payload = _load_checkpoint(path)
            if _checkpoint_stage(payload) == args.training_stage:
                compatible.append(path.resolve())
        except (KeyError, TypeError, ValueError):
            continue
    if not compatible:
        return None
    return max(compatible, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _find_automatic_local_init_checkpoint(
    args: argparse.Namespace,
) -> Path | None:
    if (
        args.training_stage != "global"
        or args.eval_only
        or args.resume is not None
        or args.init_checkpoint is not None
    ):
        return None

    checkpoint_root = _resolved_output_root(args) / "checkpoints"
    if not checkpoint_root.is_dir():
        return None

    compatible: list[Path] = []
    for path in checkpoint_root.glob("*/best_model.pt"):
        if not path.is_file():
            continue
        try:
            payload = _load_checkpoint(path)
            if _checkpoint_stage(payload) == "local":
                compatible.append(path.resolve())
        except (KeyError, TypeError, ValueError):
            continue

    if len(compatible) > 1:
        candidates = "\n".join(f"  - {path}" for path in sorted(compatible))
        raise ValueError(
            "multiple compatible local best_model.pt checkpoints were found; "
            "keep only one or select one with --init-checkpoint:\n"
            f"{candidates}"
        )
    return compatible[0] if compatible else None


def _resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name is not None:
        value = args.run_name
    elif args.resume is not None:
        value = args.resume.expanduser().resolve().parent.name
    elif args.eval_only and args.checkpoint is not None:
        value = args.checkpoint.expanduser().resolve().parent.name
    else:
        value = datetime.now().strftime(
            f"atom_pair_{args.training_stage}_%Y%m%d_%H%M%S"
        )
    return _validated_run_name(value)


def _resolve_paths(args: argparse.Namespace, run_name: str) -> PathConfig:
    base = DEFAULT_PATH_CONFIG
    output_root = _resolved_output_root(args)
    return PathConfig(
        pyg_data_dir=(args.data_dir or base.pyg_data_dir).expanduser().resolve(),
        fragment_vocab_path=(
            args.fragment_vocab or base.fragment_vocab_path
        ).expanduser().resolve(),
        atom_vocab_path=(args.atom_vocab or base.atom_vocab_path).expanduser().resolve(),
        output_root=output_root,
        checkpoint_dir=output_root / "checkpoints" / run_name,
        tensorboard_dir=output_root / "tensorboard" / run_name,
        split_manifest_path=output_root / "split_manifest.json",
        preflight_report_path=output_root / "preflight_report.json",
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _vocabulary_fingerprints(
    fragment_vocab: FragmentVocabularyLookup,
    atom_vocab: AtomVocabularyLookup,
) -> dict[str, str]:
    return {
        "fragment_vocab_sha256": fragment_vocab.sha256,
        "atom_vocab_sha256": atom_vocab.sha256,
    }


def _verify_fingerprints(
    checkpoint: Mapping[str, Any], current: Mapping[str, str]
) -> None:
    saved = checkpoint.get("vocabulary_fingerprints")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint lacks vocabulary fingerprints")
    mismatches = [
        key for key, value in current.items() if str(saved.get(key)) != str(value)
    ]
    if mismatches:
        raise ValueError(
            "checkpoint vocabulary does not match current lookup files: "
            + ", ".join(mismatches)
        )


def _checkpoint_payload(
    *,
    epoch: int,
    training_stage: str,
    init_checkpoint: Path | None,
    model: PhiSLinkerAtomPairModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: Any,
    path_config: PathConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    vocabulary_fingerprints: Mapping[str, str],
    split_manifest_sha256: str,
    best_val_total_loss: float,
    best_val_tie_metric: float,
    epochs_without_improvement: int,
    train_result: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "atom_pair_supervision_policy": ATOM_PAIR_SUPERVISION_POLICY,
        "training_stage": training_stage,
        "initial_local_checkpoint": (
            str(init_checkpoint.expanduser().resolve())
            if init_checkpoint is not None
            else None
        ),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "configuration": configuration_snapshot(
            path_config, model_config, train_config
        ),
        "vocabulary_fingerprints": dict(vocabulary_fingerprints),
        "split_manifest_sha256": str(split_manifest_sha256),
        "best_val_total_loss": float(best_val_total_loss),
        "best_val_tie_metric": float(best_val_tie_metric),
        "epochs_without_improvement": int(epochs_without_improvement),
        "train_result": dict(train_result),
        "validation_result": dict(validation_result),
        "history": list(history),
    }


def _write_tensorboard(
    writer: Any,
    epoch: int,
    train_result: Any,
    val_result: Any,
    learning_rate: float,
) -> None:
    values = {
        "loss/train_total": train_result.total_loss,
        "loss/train_local": train_result.local_loss,
        "loss/train_valid_combination": train_result.valid_combination_loss,
        "loss/train_conflict_combination": train_result.conflict_combination_loss,
        "loss/val_total": val_result.total_loss,
        "loss/val_local": val_result.local_loss,
        "loss/val_valid_combination": val_result.valid_combination_loss,
        "loss/val_conflict_combination": val_result.conflict_combination_loss,
        "metric/val_local_top1": val_result.local_top1_accuracy,
        "metric/val_top_k_truth_recall": val_result.top_k_truth_recall,
        "metric/val_global_exact": val_result.global_exact_accuracy,
        "optimizer/learning_rate": learning_rate,
    }
    for tag, value in values.items():
        if value is not None:
            writer.add_scalar(tag, float(value), epoch)


def is_better_validation_checkpoint(
    current_total_loss: float,
    current_tie_metric: float,
    best_total_loss: float,
    best_tie_metric: float,
    *,
    equality_tolerance: float = 1.0e-12,
) -> bool:
    if current_total_loss < best_total_loss - equality_tolerance:
        return True
    return (
        abs(current_total_loss - best_total_loss) <= equality_tolerance
        and current_tie_metric > best_tie_metric
    )


def _stage_limits(training_stage: str, config: TrainConfig) -> tuple[int, float, int]:
    if training_stage == "global":
        return (
            config.global_epochs,
            config.global_learning_rate,
            config.global_early_stopping_patience,
        )
    return config.epochs, config.learning_rate, config.early_stopping_patience


def _tie_metric(training_stage: str, result: Any) -> float:
    value = (
        result.global_exact_accuracy
        if training_stage == "global"
        else result.local_top1_accuracy
    )
    return float(value if value is not None else 0.0)


def main() -> None:
    args = _parse_args()
    automatic_resume = _find_automatic_resume_checkpoint(args)
    if automatic_resume is not None:
        args.resume = automatic_resume
        print(f"Auto-resume checkpoint: {automatic_resume}")
    elif (
        not args.eval_only
        and args.resume is None
        and args.init_checkpoint is None
    ):
        if args.training_stage == "global":
            automatic_init = _find_automatic_local_init_checkpoint(args)
            if automatic_init is None:
                raise ValueError(
                    "no compatible global last_model.pt or unique local "
                    "best_model.pt was found; provide "
                    "--init-checkpoint <local_best.pt>"
                )
            args.init_checkpoint = automatic_init
            print(f"Auto-selected local initialization checkpoint: {automatic_init}")
        elif not args.fresh_start:
            print(
                "Auto-resume: no compatible checkpoint found; "
                "starting local training."
            )

    run_name = _resolve_run_name(args)
    path_config = _resolve_paths(args, run_name)
    path_config.validate_inputs()
    path_config.create_output_directories()

    checkpoint_path = args.checkpoint if args.eval_only else args.resume
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    init_checkpoint = (
        _load_checkpoint(args.init_checkpoint)
        if args.init_checkpoint is not None
        else None
    )
    if checkpoint is not None and _checkpoint_stage(checkpoint) != args.training_stage:
        raise ValueError(
            "--resume/--checkpoint training_stage differs from --training-stage"
        )
    if init_checkpoint is not None and _checkpoint_stage(init_checkpoint) != "local":
        raise ValueError("--init-checkpoint must be a local-stage checkpoint")
    initial_local_checkpoint_path = args.init_checkpoint
    if initial_local_checkpoint_path is None and checkpoint is not None:
        saved_initial = checkpoint.get("initial_local_checkpoint")
        if saved_initial:
            initial_local_checkpoint_path = Path(str(saved_initial))

    config_source = checkpoint or init_checkpoint
    if config_source is None:
        model_config = DEFAULT_MODEL_CONFIG
        train_config = DEFAULT_TRAIN_CONFIG
    else:
        model_config, train_config = _checkpoint_configs(config_source)
    model_config.validate()
    train_config.validate()
    seed_everything(train_config.random_seed)
    device = _resolve_device(args.device)
    max_epochs, learning_rate, early_stopping_patience = _stage_limits(
        args.training_stage, train_config
    )

    print(f"Run: {run_name}")
    print(f"Training stage: {args.training_stage}")
    print(f"Device: {device}")
    print("Loading fixed fragment and atom vocabularies...")
    fragment_vocab = FragmentVocabularyLookup(
        path_config.fragment_vocab_path,
        expected_dim=model_config.fragment_embedding_dim,
    )
    atom_vocab = AtomVocabularyLookup(
        path_config.atom_vocab_path,
        expected_dim=model_config.atom_embedding_dim,
    )
    fingerprints = _vocabulary_fingerprints(fragment_vocab, atom_vocab)
    for payload in (checkpoint, init_checkpoint):
        if payload is not None:
            _verify_fingerprints(payload, fingerprints)

    candidate_cache = FragmentCandidateCache(atom_vocab)
    conformer_failure_registry_dir = (
        path_config.checkpoint_dir / "conformer_failure_registry"
    )
    conformer_failure_report_path = (
        path_config.checkpoint_dir / "conformer_generation_failures.json"
    )
    geometry_cache = ConformerGeometryCache(
        CONFORMER_TEMPLATE_CACHE_SIZE,
        failure_registry_dir=conformer_failure_registry_dir,
    )
    builder = ConnectionFeatureBuilder(
        fragment_vocab, candidate_cache, geometry_cache, model_config
    )

    files = discover_pyg_files(path_config.pyg_data_dir)
    print(
        f"Preflight: validating {len(files)} files, single-pair coverage, "
        "and global atom capacity..."
    )
    all_stats = run_preflight(
        files, builder, report_path=path_config.preflight_report_path
    )
    filtered_double = sum(
        item.num_filtered_double_queries for item in all_stats
    )
    zero_query_files = sum(item.num_queries == 0 for item in all_stats)
    excluded_queries = sum(item.num_excluded_queries for item in all_stats)
    print(
        "Preflight filtering: "
        f"double_queries={filtered_double}, "
        f"known_bad_queries={excluded_queries}, "
        f"files_without_retained_queries={zero_query_files}"
    )

    train_stats, validation_stats = create_or_load_split(
        all_stats,
        data_dir=path_config.pyg_data_dir,
        manifest_path=path_config.split_manifest_path,
        train_fraction=train_config.train_fraction,
        seed=train_config.random_seed,
        rebuild=args.rebuild_split,
    )
    split_hash = file_sha256(path_config.split_manifest_path)
    for payload in (checkpoint, init_checkpoint):
        if payload is None:
            continue
        saved_hash = payload.get("split_manifest_sha256")
        if saved_hash is not None and str(saved_hash) != split_hash:
            raise ValueError("current split manifest differs from the checkpoint")

    print(
        "Split: "
        f"train_graphs={len(train_stats)}, val_graphs={len(validation_stats)}, "
        f"train_queries={sum(item.num_queries for item in train_stats)}, "
        f"val_queries={sum(item.num_queries for item in validation_stats)}"
    )
    train_dataset = AtomPairGraphDataset(
        [item.path for item in train_stats],
        builder,
        include_targets=True,
        dataset_split="train",
    )
    validation_dataset = AtomPairGraphDataset(
        [item.path for item in validation_stats],
        builder,
        include_targets=True,
        dataset_split="validation",
    )
    train_loader, train_sampler = build_dataloader(
        train_dataset,
        train_stats,
        max_graphs=train_config.max_graphs_per_batch,
        max_candidates=train_config.max_candidates_per_batch,
        shuffle=True,
        seed=train_config.random_seed,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory and device.type == "cuda",
    )
    validation_loader, _ = build_dataloader(
        validation_dataset,
        validation_stats,
        max_graphs=train_config.max_graphs_per_batch,
        max_candidates=train_config.max_candidates_per_batch,
        shuffle=False,
        seed=train_config.random_seed,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory and device.type == "cuda",
    )

    model = PhiSLinkerAtomPairModel(model_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif init_checkpoint is not None:
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)

    if args.eval_only:
        result, detailed = validate_one_epoch_detailed(
            model,
            validation_loader,
            device=device,
            training_stage=args.training_stage,
            train_config=train_config,
            use_amp=train_config.use_amp,
        )
        conformer_summary = geometry_cache.write_failure_report(
            conformer_failure_report_path
        )
        report = {
            "mode": "eval_only",
            "training_stage": args.training_stage,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "validation": result.to_dict(),
            "detailed_analysis": detailed,
            "validation_graphs": len(validation_stats),
            "validation_queries": sum(item.num_queries for item in validation_stats),
            "filtered_double_queries": sum(
                item.num_filtered_double_queries for item in validation_stats
            ),
            "conformer_generation_failures": {
                "report_path": str(conformer_failure_report_path),
                "failed_smiles_count": conformer_summary["failed_smiles_count"],
                "source_occurrence_count": conformer_summary[
                    "source_occurrence_count"
                ],
            },
            "vocabulary_fingerprints": fingerprints,
            "configuration": configuration_snapshot(
                path_config, model_config, train_config
            ),
        }
        report_path = path_config.checkpoint_dir / "eval_only_report.json"
        atomic_write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Evaluation report: {report_path}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
        min_lr=train_config.min_learning_rate,
    )
    amp_enabled = train_config.use_amp and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    best_tie_metric = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_total_loss", float("inf")))
        best_tie_metric = float(
            checkpoint.get("best_val_tie_metric", float("-inf"))
        )
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = list(checkpoint.get("history", []))

    if start_epoch > max_epochs:
        print(
            f"No training epochs remain: checkpoint epoch {start_epoch - 1} "
            f"has reached the {args.training_stage} limit {max_epochs}."
        )
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("TensorBoard is required for training") from exc
    writer = SummaryWriter(log_dir=str(path_config.tensorboard_dir))
    best_path = path_config.checkpoint_dir / "best_model.pt"
    last_path = path_config.checkpoint_dir / "last_model.pt"
    history_path = path_config.checkpoint_dir / "training_history.json"
    conformer_summary: Mapping[str, Any] = {}

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            train_sampler.set_epoch(epoch)
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                training_stage=args.training_stage,
                train_config=train_config,
                epoch=epoch,
                gradient_clip_norm=train_config.gradient_clip_norm,
                use_amp=train_config.use_amp,
                scaler=scaler,
            )
            validation_result = validate_one_epoch(
                model,
                validation_loader,
                device=device,
                training_stage=args.training_stage,
                train_config=train_config,
                use_amp=train_config.use_amp,
            )
            scheduler.step(validation_result.total_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            _write_tensorboard(
                writer, epoch, train_result, validation_result, current_lr
            )
            writer.flush()
            conformer_summary = geometry_cache.write_failure_report(
                conformer_failure_report_path
            )

            tie_metric = _tie_metric(args.training_stage, validation_result)
            is_best = is_better_validation_checkpoint(
                validation_result.total_loss,
                tie_metric,
                best_val_loss,
                best_tie_metric,
            )
            if is_best:
                best_val_loss = validation_result.total_loss
                best_tie_metric = tie_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_record = {
                "epoch": epoch,
                "training_stage": args.training_stage,
                "learning_rate": current_lr,
                "train": train_result.to_dict(),
                "validation": validation_result.to_dict(),
                "failed_conformer_smiles": conformer_summary[
                    "failed_smiles_count"
                ],
                "is_best": is_best,
            }
            history.append(epoch_record)
            payload = _checkpoint_payload(
                epoch=epoch,
                training_stage=args.training_stage,
                init_checkpoint=initial_local_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                path_config=path_config,
                model_config=model_config,
                train_config=train_config,
                vocabulary_fingerprints=fingerprints,
                split_manifest_sha256=split_hash,
                best_val_total_loss=best_val_loss,
                best_val_tie_metric=best_tie_metric,
                epochs_without_improvement=epochs_without_improvement,
                train_result=train_result.to_dict(),
                validation_result=validation_result.to_dict(),
                history=history,
            )
            atomic_torch_save(last_path, payload)
            if is_best:
                atomic_torch_save(best_path, payload)
            atomic_write_json(history_path, history)

            metric_text = (
                f"global_exact={validation_result.global_exact_accuracy:.4f}"
                if args.training_stage == "global"
                else f"top1={validation_result.local_top1_accuracy:.4f}"
            )
            print(
                f"Epoch {epoch:03d} | train={train_result.total_loss:.6f} | "
                f"val={validation_result.total_loss:.6f} | {metric_text} | "
                f"lr={current_lr:.3g}{' | best' if is_best else ''}"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "Early stopping: validation total loss did not improve for "
                    f"{epochs_without_improvement} epochs."
                )
                break
    finally:
        writer.close()
        conformer_summary = geometry_cache.write_failure_report(
            conformer_failure_report_path
        )

    print(f"Best checkpoint: {best_path}")
    print(f"Lowest validation total loss: {best_val_loss:.8f}")
    print(
        f"Conformer failure report: {conformer_failure_report_path} "
        f"({conformer_summary['failed_smiles_count']} unique SMILES)"
    )


if __name__ == "__main__":
    main()
