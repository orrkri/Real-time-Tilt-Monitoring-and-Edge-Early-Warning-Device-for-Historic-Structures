# -*- coding: utf-8 -*-
"""
相机标定校正后的墙体表观位移/隆起分析脚本_结果输出修正版

功能：
1. 使用相机内参和畸变系数对所有图片先做去畸变；
2. 使用固定参照区估计并扣除整幅图像的共同运动；
3. 在墙面 ROI 内做网格化相位相关配准；
4. 输出“相机标定 + 参照区全局配准校正后”的墙面整体相对位移；
5. 同时输出“墙面整体仿射运动扣除后的局部残差”，用于判断是否为局部非均匀鼓出。

重要说明：
- corrected_wall_motion_px_median：墙面相对于固定参照区的整体表观位移，适合评价整片墙是否整体响应；
- local_residual_px_median：扣除墙面自身整体仿射运动后的局部残差，适合判断局部鼓包/非均匀变形；
- 所有 px 都是图像表观像素，不是直接的真实毫米法向隆起；
- approx_image_plane_mm 是按距离和焦距换算的图像平面近似毫米值，不等于墙面法向真实隆起量。

放置方式：
把本脚本放在与数据文件夹同一级目录。

自动识别两种数据结构：
A. 20m数据：
   无负载 / 负载1 / 负载后 / 负载2
   基准图关键词：人站在20米处
   默认 WALL_ROI = (1220, 810, 1880, 1500)

B. 20260529线下实拍25m数据：
   满载前 / 满载 / 空车 / 无车辆
   基准图关键词：25
   默认 WALL_ROI = (601, 964, 1115, 1607)

如需手动指定，请修改 MANUAL_CONFIG。
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# ===================== 相机标定参数 =====================
# 图像分辨率：2448 x 2048
# 内参矩阵 K：
CAMERA_MATRIX = np.array([
    [4.90035287e+03, 0.0, 1.18868904e+03],
    [0.0, 4.89745097e+03, 1.03354513e+03],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# 畸变系数 [k1, k2, p1, p2, k3]
DIST_COEFFS = np.array([
    [-9.98273084e-02, -3.11407213e-01, -2.18173660e-04,
     -1.58409673e-04, 1.70263913e+00]
], dtype=np.float64)

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}

# ===================== 手动配置区 =====================
# 默认 None 表示自动识别数据类型。
# 若要手动指定，取消下面示例注释并修改参数。
MANUAL_CONFIG = None

# MANUAL_CONFIG = {
#     "label": "20m",
#     "group_order": ["无负载", "负载1", "负载后", "负载2"],
#     "baseline_group": "无负载",
#     "baseline_keyword": "人站在20米处",
#     "wall_roi": (1220, 810, 1880, 1500),
#     "ref_rois": [(110, 0, 310, 1500)],
#     "distance_m": 20.0
# }

# MANUAL_CONFIG = {
#     "label": "25m",
#     "group_order": ["满载前", "满载", "空车", "无车辆"],
#     "baseline_group": "无车辆",
#     "baseline_keyword": "25",
#     "wall_roi": (601, 964, 1115, 1607),
#     "ref_rois": [(20, 80, 330, 520)],
#     "distance_m": 25.0
# }

# 网格划分
WALL_GRID_COLS = 4
WALL_GRID_ROWS = 6

# 全局配准模式：优先仿射，失败后退化为平移
USE_AFFINE_GLOBAL_ALIGNMENT = True


def imread_chinese(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def imwrite_chinese(path, img):
    path = Path(path)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图像编码失败，无法保存：{path}")
    buf.tofile(str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"图像保存失败：{path}")
    return path


def undistort_keep_size(gray):
    """
    使用标定内参和畸变系数去畸变。
    为保持 ROI 坐标基本可继续沿用，newCameraMatrix 直接使用原 CAMERA_MATRIX。
    """
    return cv2.undistort(gray, CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX)


def preprocess(gray):
    """灰度增强：CLAHE + Sobel 梯度幅值。"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.float32)


def phase_shift(img_a, img_b):
    shift, response = cv2.phaseCorrelate(img_a.astype(np.float32), img_b.astype(np.float32))
    dx, dy = shift
    return float(dx), float(dy), float(response)


def detect_config(base_dir):
    if MANUAL_CONFIG is not None:
        return MANUAL_CONFIG

    folders = {p.name for p in base_dir.iterdir() if p.is_dir()}

    if {"无负载", "负载1", "负载后", "负载2"}.issubset(folders):
        return {
            "label": "20m",
            "group_order": ["无负载", "负载1", "负载后", "负载2"],
            "baseline_group": "无负载",
            "baseline_keyword": "人站在20米处",
            "wall_roi": (1220, 810, 1880, 1500),
            # 左侧白网作为固定参照区。可根据标记图微调。
            "ref_rois": [(110, 0, 310, 1500)],
            "distance_m": 20.0
        }

    if {"满载前", "满载", "空车", "无车辆"}.issubset(folders):
        return {
            "label": "25m",
            "group_order": ["满载前", "满载", "空车", "无车辆"],
            "baseline_group": "无车辆",
            "baseline_keyword": "25",
            "wall_roi": (601, 964, 1115, 1607),
            # 左上方固定建筑/白网区域作为固定参照区。可根据标记图微调。
            "ref_rois": [(20, 80, 330, 520)],
            "distance_m": 25.0
        }

    raise RuntimeError("未能自动识别数据文件夹。请检查四个文件夹名称，或在 MANUAL_CONFIG 中手动配置。")


def list_group_images(base_dir, group_order):
    groups = {}
    for g in group_order:
        d = base_dir / g
        if d.exists() and d.is_dir():
            files = sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])
            groups[g] = files
    return groups


def choose_baseline(groups, baseline_group, baseline_keyword):
    cands = groups.get(baseline_group, [])
    for p in cands:
        if baseline_keyword in p.stem or baseline_keyword in p.name:
            return p
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"未找到基准图：{baseline_group}/{baseline_keyword}*.bmp")


def grid_cells(roi, cols, rows):
    x0, y0, x1, y1 = roi
    xs = np.linspace(x0, x1, cols + 1, dtype=int)
    ys = np.linspace(y0, y1, rows + 1, dtype=int)
    cells = []
    for r in range(rows):
        for c in range(cols):
            cells.append((r, c, (int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1]))))
    return cells


def make_ref_mask(shape, ref_rois):
    mask = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in ref_rois:
        mask[y0:y1, x0:x1] = 255
    return mask


def estimate_global_alignment(ref_prep, img_prep, ref_rois):
    """
    基于固定参照区估计当前图像到基准图的全局运动。
    优先使用 ECC 仿射配准，可同时扣除轻微平移/旋转/缩放/剪切；
    如果失败，则退化为参照区相位相关平移校正。
    返回 warp 矩阵和质量信息。
    """
    mask = make_ref_mask(ref_prep.shape, ref_rois)

    # 初始仿射矩阵：当前图像坐标 -> 基准图坐标
    if USE_AFFINE_GLOBAL_ALIGNMENT:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        try:
            cc, warp = cv2.findTransformECC(
                ref_prep.astype(np.float32),
                img_prep.astype(np.float32),
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=mask,
                gaussFiltSize=5
            )
            return warp.astype(np.float32), {
                "global_model": "ECC_AFFINE",
                "global_cc": float(cc),
                "global_ref_dx_px": float(warp[0, 2]),
                "global_ref_dy_px": float(warp[1, 2]),
                "global_ref_disp_px": float((warp[0, 2] ** 2 + warp[1, 2] ** 2) ** 0.5),
                "global_ref_response": np.nan
            }
        except Exception:
            pass

    # fallback：对所有参照 ROI 分别用 phase correlation，取中位数
    dxs, dys, reps = [], [], []
    for roi in ref_rois:
        x0, y0, x1, y1 = roi
        dx, dy, resp = phase_shift(ref_prep[y0:y1, x0:x1], img_prep[y0:y1, x0:x1])
        dxs.append(dx)
        dys.append(dy)
        reps.append(resp)

    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    # warpAffine 的矩阵用于把当前图对齐到基准图，因此平移取 -dx, -dy
    warp = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)

    return warp, {
        "global_model": "PHASE_TRANSLATION_FALLBACK",
        "global_cc": np.nan,
        "global_ref_dx_px": dx,
        "global_ref_dy_px": dy,
        "global_ref_disp_px": float((dx ** 2 + dy ** 2) ** 0.5),
        "global_ref_response": float(np.mean(reps))
    }


def warp_to_baseline(img_prep, warp):
    h, w = img_prep.shape[:2]
    return cv2.warpAffine(
        img_prep, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )


def compute_wall_grid_displacement(ref_aligned, img_aligned, wall_roi):
    """
    在全局配准校正后的图像上，计算墙面各网格剩余位移。
    这里的 dx/dy 已经扣除了固定参照区估计的全局运动。
    """
    rows = []
    dxs, dys, ds, responses = [], [], [], []

    for r, c, cell in grid_cells(wall_roi, WALL_GRID_COLS, WALL_GRID_ROWS):
        x0, y0, x1, y1 = cell
        dx, dy, resp = phase_shift(ref_aligned[y0:y1, x0:x1], img_aligned[y0:y1, x0:x1])
        disp = float((dx ** 2 + dy ** 2) ** 0.5)

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        dxs.append(dx)
        dys.append(dy)
        ds.append(disp)
        responses.append(resp)

        rows.append({
            "row": r,
            "col": c,
            "cx": cx,
            "cy": cy,
            "dx_px": dx,
            "dy_px": dy,
            "disp_px": disp,
            "response": resp
        })

    return rows, np.array(dxs), np.array(dys), np.array(ds), np.array(responses)


def fit_affine_residual(cell_rows):
    """
    在墙面网格位移场中拟合整体仿射运动，并计算局部残差。
    注意：此指标不代表整体隆起，而代表扣除整块墙面整体运动后的局部非均匀变形。
    """
    pts = []
    us = []
    for row in cell_rows:
        x = row["cx"]
        y = row["cy"]
        pts.append([x, y, 1, 0, 0, 0])
        pts.append([0, 0, 0, x, y, 1])
        us.append(row["dx_px"])
        us.append(row["dy_px"])

    A = np.array(pts, dtype=float)
    b = np.array(us, dtype=float)

    try:
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        params = np.zeros(6, dtype=float)

    residuals = []
    for row in cell_rows:
        x = row["cx"]
        y = row["cy"]
        pred_dx = params[0] * x + params[1] * y + params[2]
        pred_dy = params[3] * x + params[4] * y + params[5]
        rx = row["dx_px"] - pred_dx
        ry = row["dy_px"] - pred_dy
        rr = float((rx ** 2 + ry ** 2) ** 0.5)
        residuals.append(rr)
        row["affine_pred_dx_px"] = float(pred_dx)
        row["affine_pred_dy_px"] = float(pred_dy)
        row["local_residual_dx_px"] = float(rx)
        row["local_residual_dy_px"] = float(ry)
        row["local_residual_disp_px"] = rr

    residuals = np.array(residuals, dtype=float)
    return {
        "local_residual_px_median": float(np.median(residuals)),
        "local_residual_px_mean": float(np.mean(residuals)),
        "local_residual_px_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "local_residual_px_p95": float(np.percentile(residuals, 95)),
        "affine_params": json.dumps([round(float(v), 10) for v in params.tolist()], ensure_ascii=False)
    }


def pixel_to_approx_mm(px, distance_m):
    """
    图像平面近似换算：
    mm_per_px ≈ Z(mm) / f(px)
    这里只是图像平面横/纵向位移的近似尺度，不是法向真实隆起量。
    """
    f_mean = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2.0
    return float(px * (distance_m * 1000.0) / f_mean)


def draw_roi_check(base_gray_undistorted, base_dir, cfg):
    img = cv2.cvtColor(base_gray_undistorted, cv2.COLOR_GRAY2BGR)

    wall_roi = cfg["wall_roi"]
    x0, y0, x1, y1 = wall_roi
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)
    cv2.putText(img, f"WALL_ROI {wall_roi}", (x0 + 8, max(35, y0 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    for r, c, cell in grid_cells(wall_roi, WALL_GRID_COLS, WALL_GRID_ROWS):
        a, b, c2, d = cell
        cv2.rectangle(img, (a, b), (c2, d), (0, 255, 255), 1)
        cv2.putText(img, f"{r},{c}", (a + 5, b + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    for i, roi in enumerate(cfg["ref_rois"], start=1):
        rx0, ry0, rx1, ry1 = roi
        cv2.rectangle(img, (rx0, ry0), (rx1, ry1), (255, 0, 0), 5)
        cv2.putText(img, f"REF_ROI_{i} {roi}", (rx0 + 8, max(35, ry0 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    out_cn = base_dir / f"检测范围标记图_相机标定校正_{cfg['label']}.png"
    out_en = base_dir / f"roi_check_calibrated_corrected_{cfg['label']}.png"
    imwrite_chinese(out_cn, img)
    imwrite_chinese(out_en, img)
    print(f"已生成检测范围标记图：{out_cn}")
    print(f"同时生成英文文件名版本：{out_en}")


def main():
    base_dir = Path(__file__).resolve().parent
    cfg = detect_config(base_dir)

    groups = list_group_images(base_dir, cfg["group_order"])
    if not groups:
        raise RuntimeError("未找到图片文件夹。请将脚本放在与数据文件夹同一级目录。")

    baseline_path = choose_baseline(groups, cfg["baseline_group"], cfg["baseline_keyword"])
    baseline_gray = imread_chinese(baseline_path)
    if baseline_gray is None:
        raise RuntimeError(f"无法读取基准图：{baseline_path}")

    baseline_undist = undistort_keep_size(baseline_gray)
    baseline_prep = preprocess(baseline_undist)

    draw_roi_check(baseline_undist, base_dir, cfg)

    result_rows = []
    detail_rows = []

    for group, files in groups.items():
        for fp in files:
            gray = imread_chinese(fp)
            if gray is None:
                print(f"警告：无法读取 {fp}")
                continue

            undist = undistort_keep_size(gray)
            prep = preprocess(undist)

            warp, global_info = estimate_global_alignment(baseline_prep, prep, cfg["ref_rois"])
            aligned_prep = warp_to_baseline(prep, warp)

            cell_rows, dxs, dys, ds, responses = compute_wall_grid_displacement(
                baseline_prep, aligned_prep, cfg["wall_roi"]
            )

            # wall 相对固定参照区校正后的整体位移/隆起响应
            corrected_wall_motion_px_median = float(np.median(ds))
            corrected_wall_motion_px_mean = float(np.mean(ds))
            corrected_wall_motion_px_std = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
            corrected_mean_dx_px = float(np.mean(dxs))
            corrected_mean_dy_px = float(np.mean(dys))
            corrected_median_dx_px = float(np.median(dxs))
            corrected_median_dy_px = float(np.median(dys))
            wall_response = float(np.mean(responses))

            # 局部残差：扣除墙面自身整体仿射运动后的非均匀变形
            residual_info = fit_affine_residual(cell_rows)

            # 基准图强制为0
            is_base = (fp.name == baseline_path.name)
            if is_base:
                corrected_wall_motion_px_median = 0.0
                corrected_wall_motion_px_mean = 0.0
                corrected_wall_motion_px_std = 0.0
                corrected_mean_dx_px = 0.0
                corrected_mean_dy_px = 0.0
                corrected_median_dx_px = 0.0
                corrected_median_dy_px = 0.0
                residual_info["local_residual_px_median"] = 0.0
                residual_info["local_residual_px_mean"] = 0.0
                residual_info["local_residual_px_std"] = 0.0
                residual_info["local_residual_px_p95"] = 0.0
                global_info["global_ref_disp_px"] = 0.0
                global_info["global_ref_dx_px"] = 0.0
                global_info["global_ref_dy_px"] = 0.0

            result_rows.append({
                "group": group,
                "file": fp.name,
                "corrected_wall_motion_px_median": corrected_wall_motion_px_median,
                "corrected_wall_motion_px_mean": corrected_wall_motion_px_mean,
                "corrected_wall_motion_px_std": corrected_wall_motion_px_std,
                "corrected_mean_dx_px": corrected_mean_dx_px,
                "corrected_mean_dy_px": corrected_mean_dy_px,
                "corrected_median_dx_px": corrected_median_dx_px,
                "corrected_median_dy_px": corrected_median_dy_px,
                "corrected_wall_motion_mm_approx": pixel_to_approx_mm(corrected_wall_motion_px_median, cfg["distance_m"]),
                "local_residual_px_median": residual_info["local_residual_px_median"],
                "local_residual_px_mean": residual_info["local_residual_px_mean"],
                "local_residual_px_std": residual_info["local_residual_px_std"],
                "local_residual_px_p95": residual_info["local_residual_px_p95"],
                "local_residual_mm_approx": pixel_to_approx_mm(residual_info["local_residual_px_median"], cfg["distance_m"]),
                "wall_response": wall_response,
                **global_info,
                "affine_params_wall_residual": residual_info["affine_params"],
                "cell_disp_px": json.dumps([round(float(v), 6) for v in ds.tolist()], ensure_ascii=False),
            })

            for cell in cell_rows:
                detail_rows.append({
                    "group": group,
                    "file": fp.name,
                    **global_info,
                    **cell
                })

    df = pd.DataFrame(result_rows)
    detail_df = pd.DataFrame(detail_rows)

    label = cfg["label"]

    each_path = base_dir / f"结果_逐图_{label}.csv"
    df.to_csv(each_path, index=False, encoding="utf-8-sig")

    detail_path = base_dir / f"结果_逐网格详情_{label}.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = df.groupby("group").agg(
        n=("file", "count"),
        mean_corrected_wall_motion_px=("corrected_wall_motion_px_median", "mean"),
        std_corrected_wall_motion_px=("corrected_wall_motion_px_median", "std"),
        mean_corrected_wall_motion_mm_approx=("corrected_wall_motion_mm_approx", "mean"),
        mean_corrected_dx_px=("corrected_mean_dx_px", "mean"),
        mean_corrected_dy_px=("corrected_mean_dy_px", "mean"),
        mean_spatial_std_px=("corrected_wall_motion_px_std", "mean"),
        mean_local_residual_px=("local_residual_px_median", "mean"),
        mean_local_residual_p95_px=("local_residual_px_p95", "mean"),
        mean_local_residual_mm_approx=("local_residual_mm_approx", "mean"),
        mean_global_ref_disp_px=("global_ref_disp_px", "mean"),
        mean_wall_response=("wall_response", "mean"),
        mean_global_cc=("global_cc", "mean"),
        mean_global_ref_response=("global_ref_response", "mean"),
    ).reset_index()

    summary["order"] = summary["group"].apply(lambda x: cfg["group_order"].index(x) if x in cfg["group_order"] else 999)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary_path = base_dir / f"结果_分组汇总_{label}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 基准组时点输出
    baseline_group_df = df[df["group"] == cfg["baseline_group"]].copy()
    if cfg["label"] == "25m":
        time_order = ["车刚走", "车走10分钟", "车走25分钟"]
    else:
        time_order = ["人站在20米处"]

    baseline_group_df["time_label"] = baseline_group_df["file"].apply(lambda x: next((t for t in time_order if t in x), x))
    baseline_group_df["order"] = baseline_group_df["time_label"].apply(lambda x: time_order.index(x) if x in time_order else 999)
    baseline_group_df = baseline_group_df.sort_values(["order", "file"])

    time_path = base_dir / f"结果_基准组时点_{label}.csv"
    baseline_group_df.to_csv(time_path, index=False, encoding="utf-8-sig")

    lines = []
    lines.append(f"基准图：{baseline_path.as_posix()}")
    lines.append(f"数据类型：{label}")
    lines.append(f"墙面检测区 WALL_ROI：{cfg['wall_roi']}")
    lines.append(f"固定参照区 REF_ROIS：{cfg['ref_rois']}")
    lines.append(f"距离参数 distance_m：{cfg['distance_m']}")
    lines.append("相机标定：已使用内参矩阵和畸变系数对图像去畸变后再分析。")
    lines.append("全局校正：使用固定参照区通过 ECC 仿射配准估计当前图像到基准图的共同运动，并将当前图像对齐后再计算墙面位移。")
    lines.append("墙面响应：corrected_wall_motion_px_median = 全局校正后墙面网格位移幅值中位数。")
    lines.append("局部残差：local_residual_px_median = 再扣除墙面自身整体仿射运动后的非均匀变形残差。")
    lines.append("approx_mm 只是按距离和焦距估算的图像平面尺度，不等于法向真实隆起。")
    lines.append("")
    lines.append("【分组汇总】")
    for _, r in summary.iterrows():
        lines.append(
            f"{r['group']}: 校正后墙面整体表观位移 = {r['mean_corrected_wall_motion_px']:.6f} px，"
            f"约 {r['mean_corrected_wall_motion_mm_approx']:.3f} mm(图像平面近似)，"
            f"mean_dx = {r['mean_corrected_dx_px']:.6f} px，mean_dy = {r['mean_corrected_dy_px']:.6f} px，"
            f"空间离散度 = {r['mean_spatial_std_px']:.6f} px，"
            f"局部残差中位数 = {r['mean_local_residual_px']:.6f} px，"
            f"局部残差P95 = {r['mean_local_residual_p95_px']:.6f} px，"
            f"全局参照运动 = {r['mean_global_ref_disp_px']:.6f} px，"
            f"墙面匹配质量 = {r['mean_wall_response']:.4f}，"
            f"ECC相关系数 = {r['mean_global_cc']:.4f}"
        )

    text_path = base_dir / f"结果_摘要_{label}.txt"
    text_path.write_text("\n".join(lines), encoding="utf-8")

    # 额外输出一个极简结果表，避免用户只看到图片而忽略 CSV
    simple_path = base_dir / f"结果_核心数据_{label}.txt"
    simple_lines = []
    simple_lines.append("工况\t校正后墙面整体表观位移(px)\t局部残差中位数(px)\t全局参照运动(px)\t墙面匹配质量")
    for _, r in summary.iterrows():
        simple_lines.append(
            f"{r['group']}\t{r['mean_corrected_wall_motion_px']:.6f}\t"
            f"{r['mean_local_residual_px']:.6f}\t"
            f"{r['mean_global_ref_disp_px']:.6f}\t"
            f"{r['mean_wall_response']:.4f}"
        )
    simple_path.write_text("\n".join(simple_lines), encoding="utf-8")

    expected_outputs = [each_path, summary_path, detail_path, time_path, text_path, simple_path]
    missing = [str(p) for p in expected_outputs if (not p.exists() or p.stat().st_size == 0)]
    if missing:
        raise RuntimeError("以下结果文件没有成功生成：" + "\n".join(missing))

    print("\n分析完成。已生成以下结果文件：")
    print(f"1. 分组汇总结果：{summary_path}")
    print(f"2. 逐图结果：{each_path}")
    print(f"3. 逐网格详情：{detail_path}")
    print(f"4. 基准组时点：{time_path}")
    print(f"5. 文字摘要：{text_path}")
    print(f"6. 核心数据简表：{simple_path}")


if __name__ == "__main__":
    main()
