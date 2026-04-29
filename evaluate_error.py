"""误差评估：读取 detection_results.csv，计算误差并生成图表。"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/results/detection_results.csv")
    parser.add_argument("--out", default="data/results")
    args = parser.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["detected_mm"] in ("", None, "None"):
                continue
            rows.append({
                "filename": r["filename"],
                "face": r["face"],
                "distance_m": int(r["distance_m"]),
                "ground_truth_mm": float(r["ground_truth_mm"]),
                "detected_mm": float(r["detected_mm"]),
            })

    # 计算误差
    for r in rows:
        gt = r["ground_truth_mm"]
        det = r["detected_mm"]
        r["abs_error_mm"] = abs(det - gt)
        r["rel_error_pct"] = (r["abs_error_mm"] / gt * 100) if gt > 0 else None

    # 分组统计
    by_distance = {}
    by_displacement = {}
    for r in rows:
        d = r["distance_m"]
        by_distance.setdefault(d, []).append(r["abs_error_mm"])

        disp = r["ground_truth_mm"]
        by_displacement.setdefault(disp, []).append(r["abs_error_mm"])

    summary_lines = ["========== 误差评估汇总 ==========\n"]
    for d in sorted(by_distance.keys()):
        errors = np.array(by_distance[d])
        summary_lines.append(
            f"距离 {d}m: 样本数={len(errors)}, 平均误差={errors.mean():.2f}mm, "
            f"标准差={errors.std():.2f}mm, 最大={errors.max():.2f}mm, 最小={errors.min():.2f}mm"
        )
    for disp in sorted(by_displacement.keys()):
        errors = np.array(by_displacement[disp])
        summary_lines.append(
            f"位移 {disp}mm: 样本数={len(errors)}, 平均误差={errors.mean():.2f}mm, "
            f"标准差={errors.std():.2f}mm"
        )

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "error_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # 可视化 1: 误差-距离散点图
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in sorted(by_distance.keys()):
        subset = [r for r in rows if r["distance_m"] == d]
        x = [r["ground_truth_mm"] for r in subset]
        y = [r["abs_error_mm"] for r in subset]
        ax.scatter(x, y, label=f"{d}m", alpha=0.7)
    ax.set_xlabel("Ground Truth (mm)")
    ax.set_ylabel("Absolute Error (mm)")
    ax.set_title("Absolute Error vs Ground Truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_by_distance.png"), dpi=150)
    plt.close(fig)

    # 可视化 2: 按位移量分组的平均误差柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    displacements = sorted(by_displacement.keys())
    means = [np.mean(by_displacement[d]) for d in displacements]
    stds = [np.std(by_displacement[d]) for d in displacements]
    ax.bar([str(int(d)) for d in displacements], means, yerr=stds, capsize=4, color="steelblue")
    ax.set_xlabel("Ground Truth Displacement (mm)")
    ax.set_ylabel("Mean Absolute Error (mm)")
    ax.set_title("Mean Error by Displacement")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_by_displacement.png"), dpi=150)
    plt.close(fig)

    print(f"\n图表已保存到 {args.out}/")


if __name__ == "__main__":
    main()
