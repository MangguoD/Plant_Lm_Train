#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from plant_lm_train import DNAModel
from finetune_hier_classifier import (
    SoySegmentDataset,
    soy_collate_fn,
    HierDNAClassifier,
    select_device,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        positions = batch["positions"].to(device)
        labels = batch["label"].to(device)

        species_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=device)

        logits = model(tokens, positions, species_ids)
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def evaluate_one_task(task_file: Path, ckpt_path: Path, device, batch_size=4):
    """
    对单个 phenotype 计算 accuracy
    """
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    config = ckpt["config"]
    species_list = ckpt["species_list"]
    label_mapping = ckpt["label_mapping"]
    num_classes = len(label_mapping)
    freeze_lm = ckpt.get("freeze_lm", True)

    # 构建模型
    lm = DNAModel(config, species_list=species_list)
    model = HierDNAClassifier(lm, num_classes=num_classes, freeze_lm=freeze_lm)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # 数据
    dataset = SoySegmentDataset(task_file, config=config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=soy_collate_fn,
    )

    acc = evaluate(model, loader, device)
    return acc


def main():
    parser = argparse.ArgumentParser("Batch inference for all phenotypes")
    parser.add_argument("--data_dir", type=str, default="./downstream/data")
    parser.add_argument("--model_dir", type=str, default="./downstream/model")
    parser.add_argument("--out_fig", type=str, default="phenotype_accuracy.png")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = select_device(args.device)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    print(f"[INFO] Data dir  : {data_dir}")
    print(f"[INFO] Model dir : {model_dir}")
    print(f"[INFO] Device    : {device}")

    results = {}

    # =========================================
    # 遍历所有 phenotype.txt
    # =========================================
    for task_file in sorted(data_dir.glob("*.txt")):
        phenotype = task_file.stem
        ckpt_path = model_dir / f"{phenotype}_classifier.pth"

        if not ckpt_path.exists():
            print(f"[WARN] Skip {phenotype}: checkpoint not found")
            continue

        print(f"[INFO] Evaluating {phenotype} ...")
        acc = evaluate_one_task(
            task_file,
            ckpt_path,
            device,
            batch_size=args.batch_size,
        )

        results[phenotype] = acc
        print(f"[RESULT] {phenotype:20s} Acc = {acc:.4f}")

    if not results:
        print("[ERROR] No valid phenotype evaluated.")
        return

    # =========================================
    # 绘图
    # =========================================
    phenotypes = list(results.keys())
    accs = [results[p] for p in phenotypes]

    plt.figure(figsize=(max(8, len(phenotypes) * 0.8), 5))
    bars = plt.bar(phenotypes, accs)

    plt.ylabel("Accuracy")
    plt.xlabel("Phenotype")
    plt.ylim(0, 1.0)
    plt.title("Phenotype Classification Accuracy")

    plt.xticks(rotation=45, ha="right")

    # 在柱子上标数值
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.01,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=300)
    plt.close()

    print(f"[DONE] Figure saved to: {args.out_fig}")


if __name__ == "__main__":
    main()
