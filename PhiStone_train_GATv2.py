"""PhiSGATv2 训练启动脚本。

在项目根目录运行：python PhiStone_train_GATv2.py
数据路径、训练超参数及续训设置由 PhiSGATv2/PhiSGATv2_config.py 管理。
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 PhiSGATv2_config.py 中的配置启动活性回归训练。"
    )
    parser.parse_args()

    # 延迟导入，使 --help 无需加载模型和训练依赖。
    from PhiSGATv2.PhiSGATv2_train import main as train

    train()


if __name__ == "__main__":
    main()
