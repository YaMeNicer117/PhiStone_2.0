"""PhiSSE3TD 基础训练与微调启动脚本。

在项目根目录运行（单卡同样使用 torchrun）：
    torchrun --standalone --nproc_per_node=1 PhiStone_train_SE3TD.py --mode pretrain
    torchrun --standalone --nproc_per_node=1 PhiStone_train_SE3TD.py --mode creative

其他微调模式：survival_linker、survival_modify、activity_linker、activity_modify。
多卡训练调整 --nproc_per_node；训练超参数仍由原配置/微调脚本管理。
--pretrained 默认使用 diffusion_pretrain/best_model.pt；指定相对文件名时，
相对于 diffusion_pretrain 目录解析，也可提供该目录内权重的绝对路径。
保留原训练脚本的断点续训行为。
"""

import argparse
import importlib
import os
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "PhiSSE3TD"
PRETRAIN_DIR = MODEL_DIR / "training_artifacts" / "checkpoints" / "diffusion_pretrain"
TRAINING_MODES = (
    "pretrain",
    "creative",
    "survival_linker",
    "survival_modify",
    "activity_linker",
    "activity_modify",
)


def build_finetune_settings(mode: str, pretrained: Path):
    """复用原超参数，只覆盖模式、来源权重及模式相关的输出路径。"""
    family = mode.split("_", 1)[0]
    module = importlib.import_module(f"PhiSSE3TD_train_finetune_{family}")
    settings = module.build_settings()
    changes = {"source_checkpoint_path": pretrained}

    if family == "survival":
        # survival 的拓扑开关在原脚本中最终体现为 strategy。
        checkpoint_dir = settings.checkpoint_dir.with_name(mode)
        changes.update(
            strategy=mode,
            checkpoint_dir=checkpoint_dir,
            best_checkpoint_path=checkpoint_dir / f"best_model_{mode}.pt",
            last_checkpoint_path=checkpoint_dir / f"last_model_{mode}.pt",
            training_profile_path=checkpoint_dir / f"training_data_profile_{mode}.json",
            tensorboard_dir=settings.tensorboard_dir.with_name(mode),
        )
    elif family == "activity":
        # 两种 activity 模式共享内部策略，使用拓扑开关区分。
        # 沿用原 activity 目录，以带模式名的文件和日志子目录区分产物。
        checkpoint_dir = settings.checkpoint_dir
        changes.update(
            strategy="activity",
            activity_require_linker_topology=(mode == "activity_linker"),
            best_checkpoint_path=checkpoint_dir / f"best_model_{mode}.pt",
            last_checkpoint_path=checkpoint_dir / f"last_model_{mode}.pt",
            training_profile_path=checkpoint_dir / f"training_data_profile_{mode}.json",
            tensorboard_dir=settings.tensorboard_dir.with_name(mode),
        )

    return replace(settings, **changes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通过 torchrun 启动 PhiSSE3TD 基础训练或指定模式的微调。"
    )
    parser.add_argument("--mode", choices=TRAINING_MODES, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="微调来源权重：diffusion_pretrain 内的文件名或绝对路径，默认 best_model.pt。",
    )
    args = parser.parse_args()

    pretrained = None
    if args.mode == "pretrain":
        if args.pretrained is not None:
            parser.error("--pretrained 仅用于微调；基础训练沿用原脚本的续训设置。")
    else:
        pretrained = args.pretrained or Path("best_model.pt")
        if not pretrained.is_absolute():
            pretrained = PRETRAIN_DIR / pretrained
        pretrained = pretrained.resolve()
        if pretrained.parent != PRETRAIN_DIR.resolve():
            parser.error(f"微调来源权重必须位于目录：{PRETRAIN_DIR}")
        if not pretrained.is_file():
            parser.error(f"预训练权重不存在：{pretrained}。请先完成 pretrain 基础训练。")

    if "LOCAL_RANK" not in os.environ:
        parser.error(
            "请使用 torchrun 启动，例如：torchrun --standalone --nproc_per_node=1 "
            f"PhiStone_train_SE3TD.py --mode {args.mode}"
        )

    # 原 SE3TD 脚本通过同目录模块名导入，保持其现有导入方式。
    sys.path.insert(0, str(MODEL_DIR))
    if args.mode == "pretrain":
        from PhiSSE3TD_train_pretrain import main as train

        train()
        return

    settings = build_finetune_settings(args.mode, pretrained)
    if args.mode.startswith("activity_"):
        activity_checkpoint = settings.activity_checkpoint_path
        if activity_checkpoint is None or not activity_checkpoint.is_file():
            parser.error(
                f"PhiSGATv2 活性指导权重不存在：{activity_checkpoint}。"
                "请先训练 PhiSGATv2，或在 activity 微调脚本中配置已有权重。"
            )

    if os.environ.get("RANK", "0") == "0":
        print(f"训练模式：{args.mode}", flush=True)
        print(f"预训练权重：{settings.source_checkpoint_path}", flush=True)
        print(f"微调权重目录：{settings.checkpoint_dir}", flush=True)

    from PhiSSE3TD_train_finetune_common import run_finetuning

    run_finetuning(settings)


if __name__ == "__main__":
    main()
