"""
从 TensorBoard 日志中读取标量并直接画图为 PNG 图片。

用法示例：
    python tb_export_and_plot.py --logdir tb_logs --out_dir tb_plots
"""

import os
import glob
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_scalars_from_event_file(event_file: str):
    """
    从单个 TensorBoard event 文件中读取所有 scalar 序列。

    返回：
        dict: { tag: {"steps": [...], "values": [...]} }
    """
    print(f"[INFO] Loading event file: {event_file}")
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 = 读全量
        }
    )
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if not scalar_tags:
        print(f"[WARN] No scalar data found in {event_file}")
        return {}

    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": steps, "values": values}

    return data


def plot_single_tag(tag: str, steps, values, out_dir: Path):
    """
    为单个 tag 画一张曲线图。
    """
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{safe_tag}.png"

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("step")
    plt.ylabel(tag)
    plt.title(tag)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved scalar tag '{tag}' -> {out_path}")


def plot_train_val_loss_combined(data: dict, out_dir: Path):
    """
    如果存在 train/loss 和 val/token_loss，则额外画一张对比图。
    """
    train_tag = "train/loss"
    val_tag = "val/token_loss"

    if train_tag not in data or val_tag not in data:
        return

    t_steps = data[train_tag]["steps"]
    t_vals = data[train_tag]["values"]

    v_steps = data[val_tag]["steps"]
    v_vals = data[val_tag]["values"]

    out_path = out_dir / "train_vs_val_loss.png"

    plt.figure()
    plt.plot(t_steps, t_vals, label=train_tag)
    plt.plot(v_steps, v_vals, label=val_tag)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[PLOT] Saved combined loss plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="tb_logs",
        help="TensorBoard 日志目录（包含 events.out.tfevents.* 文件）"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="tb_plots",
        help="输出图片的目录"
    )
    parser.add_argument(
        "--use_latest",
        action="store_true",
        help="如指定，则只使用最新的一个 event 文件；否则处理所有找到的文件"
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out_dir)

    if not logdir.exists():
        raise FileNotFoundError(f"logdir not found: {logdir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 递归搜索所有 event 文件
    pattern = str(logdir / "**" / "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    if not event_files:
        # 有些版本文件名可能是 event*
        pattern2 = str(logdir / "**" / "event*")
        event_files = glob.glob(pattern2, recursive=True)

    if not event_files:
        raise RuntimeError(f"No event files found under {logdir}")

    # 只用最新一个文件（可选）
    if args.use_latest:
        event_files = [max(event_files, key=os.path.getmtime)]
        print(f"[INFO] Using latest event file only: {event_files[0]}")
    else:
        print(f"[INFO] Found {len(event_files)} event files, will process all.")

    # 简单做法：把所有 event 文件的同名 tag 拼接到一起
    merged_data = {}  # {tag: {"steps": [...], "values": [...]}}

    for ef in sorted(event_files):
        data = load_scalars_from_event_file(ef)
        for tag, series in data.items():
            if tag not in merged_data:
                merged_data[tag] = {"steps": [], "values": []}
            merged_data[tag]["steps"].extend(series["steps"])
            merged_data[tag]["values"].extend(series["values"])

    if not merged_data:
        print("[WARN] No scalar data merged, nothing to plot.")
        return

    # 为每个 tag 画一张单独的图
    for tag, series in merged_data.items():
        # 按 step 排序，避免多文件拼接时顺序混乱
        steps, values = zip(*sorted(zip(series["steps"], series["values"]), key=lambda x: x[0]))
        plot_single_tag(tag, steps, values, out_dir)

    # 额外画 train vs val loss 对比（如果存在）
    plot_train_val_loss_combined(merged_data, out_dir)

    print(f"[DONE] All plots saved under: {out_dir}")


if __name__ == "__main__":
    main()