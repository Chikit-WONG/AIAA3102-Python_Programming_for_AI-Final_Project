"""
Loss Curve 可视化脚本
用于绘制训练过程中的 loss 曲线
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def load_jsonl_logs(filepath: str) -> List[Dict]:
    """从 JSONL 文件加载训练日志"""
    logs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def extract_loss_from_logs(logs: List[Dict]) -> Dict[str, List]:
    """从训练日志中提取 loss 数据"""
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []

    for log in logs:
        # 解析 global_step
        if "global_step/max_steps" in log:
            step_str = log["global_step/max_steps"]
            current_step = int(step_str.split("/")[0])
        else:
            continue

        # 提取训练 loss
        if "loss" in log and "eval_loss" not in log:
            train_loss.append(log["loss"])
            train_steps.append(current_step)

        # 提取验证 loss
        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])
            eval_steps.append(current_step)

    return {
        "train_loss": train_loss,
        "train_steps": train_steps,
        "eval_loss": eval_loss,
        "eval_steps": eval_steps,
    }


def plot_loss_curve(
    log_filepath: str,
    output_path: Optional[str] = None,
    title: str = "Training Loss Curve",
):
    """
    绘制 loss curve

    Args:
        log_filepath: JSONL 日志文件路径
        output_path: 输出图片路径（可选）
        title: 图表标题
    """
    # 加载日志
    if not os.path.exists(log_filepath):
        print(f"Error: Log file not found: {log_filepath}")
        return

    logs = load_jsonl_logs(log_filepath)
    if not logs:
        print("Error: No logs found in file")
        return

    # 提取 loss 数据
    loss_data = extract_loss_from_logs(logs)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制训练 loss
    if loss_data["train_loss"]:
        ax.plot(
            loss_data["train_steps"],
            loss_data["train_loss"],
            "b-",
            label="Training Loss",
            alpha=0.7,
            linewidth=1.5,
        )

    # 绘制验证 loss
    if loss_data["eval_loss"]:
        ax.plot(
            loss_data["eval_steps"],
            loss_data["eval_loss"],
            "r-",
            label="Validation Loss",
            alpha=0.9,
            linewidth=2,
            marker="o",
            markersize=5,
        )

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或显示图表
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Loss curve saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_loss_curves(
    log_files: Dict[str, str],
    output_path: Optional[str] = None,
    title: str = "Loss Curve Comparison",
):
    """
    绘制多个模型的 loss curve 对比图

    Args:
        log_files: {模型名称: 日志文件路径} 的字典
        output_path: 输出图片路径（可选）
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (name, log_filepath) in enumerate(log_files.items()):
        if not os.path.exists(log_filepath):
            print(f"Warning: Log file not found for {name}: {log_filepath}")
            continue

        logs = load_jsonl_logs(log_filepath)
        loss_data = extract_loss_from_logs(logs)
        color = colors[idx % len(colors)]

        # 训练 loss
        if loss_data["train_loss"]:
            ax1.plot(
                loss_data["train_steps"],
                loss_data["train_loss"],
                color=color,
                label=name,
                alpha=0.7,
                linewidth=1.5,
            )

        # 验证 loss
        if loss_data["eval_loss"]:
            ax2.plot(
                loss_data["eval_steps"],
                loss_data["eval_loss"],
                color=color,
                label=name,
                alpha=0.9,
                linewidth=2,
                marker="o",
                markersize=5,
            )

    ax1.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Training Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comparison loss curve saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """主函数"""
    # 日志文件路径
    task1_log = "./logging_t1.jsonl"
    task2_log = "./logging_t2.jsonl"

    # 确保输出目录存在
    os.makedirs("./eval_output", exist_ok=True)

    # 绘制 Task 1 的 loss curve
    if os.path.exists(task1_log):
        plot_loss_curve(
            log_filepath=task1_log,
            output_path="./eval_output/task1_loss_curve.png",
            title="Task 1: Intent Classification Loss Curve",
        )
        print("✓ Task 1 loss curve generated")
    else:
        print(f"Task 1 log not found: {task1_log}")

    # 绘制 Task 2 的 loss curve
    if os.path.exists(task2_log):
        plot_loss_curve(
            log_filepath=task2_log,
            output_path="./eval_output/task2_loss_curve.png",
            title="Task 2: Response Generation Loss Curve",
        )
        print("✓ Task 2 loss curve generated")
    else:
        print(f"Task 2 log not found: {task2_log}")

    # 对比图（如果两个日志都存在）
    if os.path.exists(task1_log) and os.path.exists(task2_log):
        plot_comparison_loss_curves(
            log_files={
                "Task 1 (Classification)": task1_log,
                "Task 2 (Response Generation)": task2_log,
            },
            output_path="./eval_output/comparison_loss_curve.png",
            title="Training Loss Comparison: Task 1 vs Task 2",
        )
        print("✓ Comparison loss curve generated")

    print("\nLoss curve visualization complete!")


if __name__ == "__main__":
    main()
