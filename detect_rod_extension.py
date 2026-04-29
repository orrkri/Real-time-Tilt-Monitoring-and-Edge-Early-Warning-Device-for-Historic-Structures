"""主检测脚本：遍历 1面/2面 所有图片，检测木棒延伸量并输出 CSV。"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from utils.calibration import load_calibration, undistort_image


def imread_safe(path):
    """兼容中文路径的图片读取。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

from utils.rod_detector import detect_rod_extension_pixels


def parse_filename(filename: str) -> dict:
    """解析文件名，提取距离和真实延伸量。

    支持格式:
      - 10m0mm1.bmp  → distance=10, ground_truth_mm=0
      - 10m5mm1.bmp  → distance=10, ground_truth_mm=5
      - 10m1cm1.bmp  → distance=10, ground_truth_mm=10
      - 10m2cm1.bmp  → distance=10, ground_truth_mm=20
      - 10m01.bmp    → distance=10, ground_truth_mm=0 (简写)
      - empty1.bmp   → distance=None, ground_truth_mm=0
      - 1mruler1.bmp → ruler 图
    """
    name = Path(filename).stem.lower()
    result = {"distance_m": None, "ground_truth_mm": None, "is_ruler": False, "is_empty": False}

    if "ruler" in name:
        result["is_ruler"] = True
        m = re.search(r'(\d+)m', name)
        if m:
            result["distance_m"] = int(m.group(1))
        return result

    if "empty" in name:
        result["is_empty"] = True
        result["ground_truth_mm"] = 0
        m = re.search(r'(\d+)m', name)
        if m:
            result["distance_m"] = int(m.group(1))
        return result

    # 标准格式: 10m0mm1, 10m5mm1, 10m1cm1, 10m2cm1, 10m5mm04
    m = re.match(r'(\d+)m(\d+)(mm|cm)(\d+)?', name)
    if m:
        result["distance_m"] = int(m.group(1))
        val = int(m.group(2))
        unit = m.group(3)
        if unit == "cm":
            result["ground_truth_mm"] = val * 10
        else:
            result["ground_truth_mm"] = val
        return result

    # 简写格式: 10m01, 10m02, 20m01 → 位移为 0
    m = re.match(r'(\d+)m(\d+)$', name)
    if m:
        result["distance_m"] = int(m.group(1))
        short = m.group(2)
        # 如果以 0 开头（01, 02），视为 0mm 的简写
        if short.startswith('0') and len(short) <= 2:
            result["ground_truth_mm"] = 0
        else:
            result["ground_truth_mm"] = int(short)
        return result

    return result


def compute_scale_k(fx: float, distance_m: int) -> float:
    """根据相机焦距和拍摄距离计算像素到毫米的转换系数。

    基于小孔成像模型：physical_size = pixel_size * distance / fx
    因此 k = distance(mm) / fx(px) = mm/px
    """
    return (distance_m * 1000.0) / fx


def main():
    parser = argparse.ArgumentParser(description="检测木棒延伸量")
    parser.add_argument("--calib", default="calibration_result.yaml", help="标定文件路径")
    parser.add_argument("--out", default="data/results/detection_results.csv", help="输出 CSV 路径")
    parser.add_argument("--face", default="1面", choices=["1面", "2面"], help="检测哪个面")
    parser.add_argument("--show", action="store_true", help="显示每步可视化结果（调试用）")
    args = parser.parse_args()

    # 1. 加载标定参数
    print("[1/4] 加载标定参数...")
    K, D, _ = load_calibration(args.calib)
    fx = K[0, 0]

    # 2. 读取图片列表
    face_dir = args.face
    image_paths = sorted(Path(face_dir).glob("*.bmp"))
    if not image_paths:
        raise RuntimeError(f"在 {face_dir} 中未找到 .bmp 图片")

    # 3. 按距离分组，每组内找到 0mm 基线
    by_distance = defaultdict(list)
    for p in image_paths:
        info = parse_filename(p.name)
        if info["is_ruler"]:
            continue
        dist = info.get("distance_m")
        if dist is None:
            continue
        by_distance[dist].append((p, info))

    if not by_distance:
        raise RuntimeError("未找到可处理的测试图片")

    print(f"[2/4] 找到 {len(by_distance)} 个距离组: {sorted(by_distance.keys())}")

    # 4. 去畸变基线并缓存
    baseline_cache = {}
    for dist in sorted(by_distance.keys()):
        items = by_distance[dist]
        baseline_candidates = [(p, info) for p, info in items if info["ground_truth_mm"] == 0]
        if not baseline_candidates:
            # 回退：寻找同距离的 empty 图
            empty_candidates = [
                p for p in image_paths
                if parse_filename(p.name)["is_empty"] and parse_filename(p.name).get("distance_m") == dist
            ]
            if empty_candidates:
                baseline_path = empty_candidates[0]
                print(f"  距离 {dist}m: 使用 empty 图作为基线 ({baseline_path.name})")
            else:
                print(f"  WARN: 距离 {dist}m 缺少 0mm 基线，跳过该组")
                continue
        else:
            baseline_path = baseline_candidates[0][0]
            if len(baseline_candidates) > 1:
                # 多张基线时取平均
                imgs = [undistort_image(imread_safe(p), K, D).astype(np.float32) for p, _ in baseline_candidates]
                baseline_cache[dist] = np.mean(imgs, axis=0).astype(np.uint8)
                print(f"  距离 {dist}m: 平均 {len(baseline_candidates)} 张基线图")
            else:
                baseline_cache[dist] = undistort_image(imread_safe(baseline_path), K, D)
                print(f"  距离 {dist}m: 基线 {baseline_path.name}")

    print(f"[3/4] fx={fx:.2f}, 理论比例系数: 10m={compute_scale_k(fx,10):.4f} mm/px, 18m={compute_scale_k(fx,18):.4f} mm/px")

    # 5. 遍历检测
    results = []
    total = sum(len(items) for dist, items in by_distance.items() if dist in baseline_cache)
    print(f"[4/4] 开始检测 {total} 张图片...")

    for dist in sorted(by_distance.keys()):
        if dist not in baseline_cache:
            continue

        baseline_img = baseline_cache[dist]
        k = compute_scale_k(fx, dist)
        items = by_distance[dist]

        for p, info in items:
            if info["ground_truth_mm"] == 0:
                continue  # 跳过基线本身

            gt = info.get("ground_truth_mm")
            img = undistort_image(imread_safe(p), K, D)

            try:
                ext_px, tip_x, mask = detect_rod_extension_pixels(
                    img, baseline_img, face=args.face
                )
                detected_mm = ext_px * k
            except RuntimeError as e:
                print(f"  WARN {p.name}: detection failed - {e}")
                ext_px, tip_x, detected_mm = -1, -1, -1

            results.append({
                "filename": p.name,
                "face": args.face,
                "distance_m": dist,
                "ground_truth_mm": gt,
                "detected_mm": round(detected_mm, 2) if detected_mm >= 0 else None,
                "extension_pixels": ext_px,
                "tip_x": tip_x,
            })

            if detected_mm >= 0:
                print(f"  OK {p.name}: gt={gt}mm, detected={detected_mm:.2f}mm")
            else:
                print(f"  FAIL {p.name}: detection failed")

    # 6. 保存 CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "face", "distance_m", "ground_truth_mm",
            "detected_mm", "extension_pixels", "tip_x"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
