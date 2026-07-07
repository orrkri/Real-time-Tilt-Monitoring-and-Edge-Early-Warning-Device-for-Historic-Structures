# -*- coding: utf-8 -*-
import base64
import json

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


app = Flask(__name__, static_folder=".", static_url_path="", template_folder=".")


# ===================== 相机标定参数：来自原始严谨版脚本 =====================
# 图像分辨率：2448 x 2048
DEFAULT_CAMERA_MATRIX = np.array([
    [4.90035287e+03, 0.0, 1.18868904e+03],
    [0.0, 4.89745097e+03, 1.03354513e+03],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DEFAULT_DIST_COEFFS = np.array([
    [-9.98273084e-02, -3.11407213e-01, -2.18173660e-04,
     -1.58409673e-04, 1.70263913e+00]
], dtype=np.float64)

USE_AFFINE_GLOBAL_ALIGNMENT = True
DEFAULT_DISTANCE_M = 25.0


def decode_image(file_storage):
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise ValueError(f"无法读取图片：{file_storage.filename}")
    return gray, color


def parse_camera_params(raw_text):
    if not raw_text:
        return DEFAULT_CAMERA_MATRIX.copy(), DEFAULT_DIST_COEFFS.copy(), {
            "fx": float(DEFAULT_CAMERA_MATRIX[0, 0]),
            "fy": float(DEFAULT_CAMERA_MATRIX[1, 1]),
            "cx": float(DEFAULT_CAMERA_MATRIX[0, 2]),
            "cy": float(DEFAULT_CAMERA_MATRIX[1, 2]),
            "k1": float(DEFAULT_DIST_COEFFS[0, 0]),
            "k2": float(DEFAULT_DIST_COEFFS[0, 1]),
            "p1": float(DEFAULT_DIST_COEFFS[0, 2]),
            "p2": float(DEFAULT_DIST_COEFFS[0, 3]),
            "k3": float(DEFAULT_DIST_COEFFS[0, 4]),
        }

    params = json.loads(raw_text)
    defaults = {
        "fx": float(DEFAULT_CAMERA_MATRIX[0, 0]),
        "fy": float(DEFAULT_CAMERA_MATRIX[1, 1]),
        "cx": float(DEFAULT_CAMERA_MATRIX[0, 2]),
        "cy": float(DEFAULT_CAMERA_MATRIX[1, 2]),
        "k1": float(DEFAULT_DIST_COEFFS[0, 0]),
        "k2": float(DEFAULT_DIST_COEFFS[0, 1]),
        "p1": float(DEFAULT_DIST_COEFFS[0, 2]),
        "p2": float(DEFAULT_DIST_COEFFS[0, 3]),
        "k3": float(DEFAULT_DIST_COEFFS[0, 4]),
    }
    values = {}
    for key, fallback in defaults.items():
        value = params.get(key, fallback)
        try:
            values[key] = float(value)
        except Exception as exc:
            raise ValueError(f"相机参数 {key} 不是有效数字。") from exc

    if values["fx"] <= 0 or values["fy"] <= 0:
        raise ValueError("fx/fy 必须大于 0。")

    camera_matrix = np.array([
        [values["fx"], 0.0, values["cx"]],
        [0.0, values["fy"], values["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.array([[
        values["k1"], values["k2"], values["p1"], values["p2"], values["k3"]
    ]], dtype=np.float64)
    return camera_matrix, dist_coeffs, values


def undistort_keep_size(gray, camera_matrix, dist_coeffs):
    """
    使用标定内参和畸变系数去畸变。
    为保持 ROI 坐标基本可继续沿用，newCameraMatrix 直接使用原 CAMERA_MATRIX。
    """
    return cv2.undistort(gray, camera_matrix, dist_coeffs, None, camera_matrix)


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


def clamp_roi(roi, width, height, label):
    x0 = int(max(0, min(width - 1, round(float(roi["x0"])))))
    y0 = int(max(0, min(height - 1, round(float(roi["y0"])))))
    x1 = int(max(1, min(width, round(float(roi["x1"])))))
    y1 = int(max(1, min(height, round(float(roi["y1"])))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{label}无效，请重新框选。")
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError(f"{label}过小，请重新框选。")
    return x0, y0, x1, y1


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
    """
    mask = make_ref_mask(ref_prep.shape, ref_rois)

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
                "global_ref_response": None,
            }
        except Exception:
            pass

    dxs, dys, reps = [], [], []
    for roi in ref_rois:
        x0, y0, x1, y1 = roi
        dx, dy, resp = phase_shift(ref_prep[y0:y1, x0:x1], img_prep[y0:y1, x0:x1])
        dxs.append(dx)
        dys.append(dy)
        reps.append(resp)

    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    warp = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)

    return warp, {
        "global_model": "PHASE_TRANSLATION_FALLBACK",
        "global_cc": None,
        "global_ref_dx_px": dx,
        "global_ref_dy_px": dy,
        "global_ref_disp_px": float((dx ** 2 + dy ** 2) ** 0.5),
        "global_ref_response": float(np.mean(reps)),
    }


def warp_to_baseline(img_prep, warp):
    h, w = img_prep.shape[:2]
    return cv2.warpAffine(
        img_prep, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )


def compute_wall_grid_displacement(ref_aligned, img_aligned, wall_roi, cols, rows):
    """
    在全局配准校正后的图像上，计算墙面各网格剩余位移。
    这里的 dx/dy 已经扣除了固定参照区估计的全局运动。
    """
    cell_rows = []
    dxs, dys, ds, responses = [], [], [], []

    for r, c, cell in grid_cells(wall_roi, cols, rows):
        x0, y0, x1, y1 = cell
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError("墙体分析区分块过小，请减少行列数或框选更大的区域。")

        dx, dy, resp = phase_shift(ref_aligned[y0:y1, x0:x1], img_aligned[y0:y1, x0:x1])
        disp = float((dx ** 2 + dy ** 2) ** 0.5)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        dxs.append(dx)
        dys.append(dy)
        ds.append(disp)
        responses.append(resp)
        cell_rows.append({
            "row": r,
            "col": c,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "cx": cx,
            "cy": cy,
            "dx_px": dx,
            "dy_px": dy,
            "disp_px": disp,
            "response": resp,
        })

    return cell_rows, np.array(dxs), np.array(dys), np.array(ds), np.array(responses)


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
        "affine_params": [round(float(v), 10) for v in params.tolist()],
    }


def pixel_to_approx_mm(px, distance_m, camera_matrix):
    f_mean = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0
    return float(px * (distance_m * 1000.0) / f_mean)


def mark_prominent_cells(cell_rows):
    residuals = np.array([c["local_residual_disp_px"] for c in cell_rows], dtype=float)
    disps = np.array([c["disp_px"] for c in cell_rows], dtype=float)
    residual_threshold = float(np.percentile(residuals, 85))
    top_count = max(1, int(np.ceil(len(cell_rows) * 0.12)))
    top_indices = set(np.argsort(residuals)[-top_count:].tolist())
    median_disp = float(np.median(disps))

    for idx, row in enumerate(cell_rows):
        row["prominence_px"] = float(row["disp_px"] - median_disp)
        row["is_prominent"] = bool(
            (row["local_residual_disp_px"] >= residual_threshold and row["local_residual_disp_px"] > 0)
            or idx in top_indices
        )

    return residual_threshold


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def color_for_cell(value, min_value, max_value, prominent):
    t = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0
    if prominent:
        return np.array([40, 30, 230], dtype=np.float32), 0.30 + 0.30 * t
    return np.array([95, 120, 20], dtype=np.float32), 0.14 + 0.22 * t


def draw_annotation(target_color_undist, wall_roi, ref_rois, cells, summary):
    out = target_color_undist.copy()
    residual_values = [c["local_residual_disp_px"] for c in cells]
    min_residual = min(residual_values)
    max_residual = max(residual_values)

    for idx, roi in enumerate(ref_rois, start=1):
        x0, y0, x1, y1 = roi
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 5)
        cv2.putText(
            out, f"REF_{idx}", (x0 + 8, max(30, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA
        )

    wx0, wy0, wx1, wy1 = wall_roi
    cv2.rectangle(out, (wx0, wy0), (wx1, wy1), (0, 0, 255), 5)
    cv2.putText(
        out, "WALL_ROI", (wx0 + 8, max(35, wy0 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA
    )

    for cell in cells:
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        color, alpha = color_for_cell(
            cell["local_residual_disp_px"],
            min_residual,
            max_residual,
            cell["is_prominent"],
        )
        patch = out[y0:y1, x0:x1].astype(np.float32)
        patch = patch * (1.0 - alpha) + color * alpha
        out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

        border = (0, 0, 255) if cell["is_prominent"] else (0, 215, 255)
        thickness = 4 if cell["is_prominent"] else 1
        cv2.rectangle(out, (x0, y0), (x1, y1), border, thickness)

        cx = int(cell["cx"])
        cy = int(cell["cy"])
        ex = int(cx + cell["dx_px"] * 8)
        ey = int(cy + cell["dy_px"] * 8)
        cv2.arrowedLine(out, (cx, cy), (ex, ey), (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(
            out,
            f"{cell['local_residual_disp_px']:.2f}",
            (x0 + 5, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255) if cell["is_prominent"] else (20, 30, 40),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(out, (18, 18), (760, 164), (255, 255, 255), -1)
    cv2.rectangle(out, (18, 18), (760, 164), (40, 40, 40), 1)
    lines = [
        f"Predicted bulge: {summary['predicted_bulge_mm_approx']:.3f} mm",
        f"Corrected wall motion median: {summary['corrected_wall_motion_px_median']:.4f} px",
        f"Local residual P95: {summary['local_residual_px_p95']:.4f} px",
        f"Global ref motion: {summary['global_ref_disp_px']:.4f} px  Model: {summary['global_model']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (36, 54 + i * 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 30, 40),
            2, cv2.LINE_AA
        )
    return out


def png_base64(img):
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("标注图编码失败。")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "base_image" not in request.files or "target_image" not in request.files:
            raise ValueError("请上传基准图和待计算图。")

        cols = max(2, min(16, int(request.form.get("cols", 4))))
        rows = max(2, min(16, int(request.form.get("rows", 6))))
        distance_m = float(request.form.get("distance_m", DEFAULT_DISTANCE_M))
        camera_matrix, dist_coeffs, camera_params = parse_camera_params(request.form.get("camera_params", ""))
        wall_raw = json.loads(request.form.get("wall_roi", "{}"))
        ref_raw = json.loads(request.form.get("ref_rois", "[]"))
        if not ref_raw:
            raise ValueError("请至少框选一个固定参考区。")

        ref_gray, _ = decode_image(request.files["base_image"])
        target_gray, target_color = decode_image(request.files["target_image"])
        if ref_gray.shape != target_gray.shape:
            raise ValueError("两张图尺寸不一致，无法逐块对比。")

        height, width = ref_gray.shape[:2]
        wall_roi = clamp_roi(wall_raw, width, height, "墙体分析区")
        ref_rois = [clamp_roi(roi, width, height, f"固定参考区{i + 1}") for i, roi in enumerate(ref_raw)]

        baseline_undist = undistort_keep_size(ref_gray, camera_matrix, dist_coeffs)
        target_undist = undistort_keep_size(target_gray, camera_matrix, dist_coeffs)
        target_color_undist = cv2.undistort(target_color, camera_matrix, dist_coeffs, None, camera_matrix)

        baseline_prep = preprocess(baseline_undist)
        target_prep = preprocess(target_undist)

        warp, global_info = estimate_global_alignment(baseline_prep, target_prep, ref_rois)
        aligned_prep = warp_to_baseline(target_prep, warp)

        cell_rows, dxs, dys, ds, responses = compute_wall_grid_displacement(
            baseline_prep, aligned_prep, wall_roi, cols, rows
        )
        residual_info = fit_affine_residual(cell_rows)
        prominent_threshold = mark_prominent_cells(cell_rows)

        corrected_wall_motion_px_median = float(np.median(ds))
        corrected_wall_motion_px_mean = float(np.mean(ds))
        corrected_wall_motion_px_std = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        local_residual_px_median = residual_info["local_residual_px_median"]
        predicted_bulge_px = residual_info["local_residual_px_p95"]
        predicted_bulge_mm_approx = pixel_to_approx_mm(predicted_bulge_px, distance_m, camera_matrix)

        summary = {
            "predicted_bulge_px": predicted_bulge_px,
            "predicted_bulge_mm_approx": predicted_bulge_mm_approx,
            "corrected_wall_motion_px_median": corrected_wall_motion_px_median,
            "corrected_wall_motion_px_mean": corrected_wall_motion_px_mean,
            "corrected_wall_motion_px_std": corrected_wall_motion_px_std,
            "corrected_mean_dx_px": float(np.mean(dxs)),
            "corrected_mean_dy_px": float(np.mean(dys)),
            "corrected_median_dx_px": float(np.median(dxs)),
            "corrected_median_dy_px": float(np.median(dys)),
            "corrected_wall_motion_mm_approx": pixel_to_approx_mm(corrected_wall_motion_px_median, distance_m, camera_matrix),
            "local_residual_px_median": local_residual_px_median,
            "local_residual_px_mean": residual_info["local_residual_px_mean"],
            "local_residual_px_std": residual_info["local_residual_px_std"],
            "local_residual_px_p95": residual_info["local_residual_px_p95"],
            "local_residual_mm_approx": pixel_to_approx_mm(local_residual_px_median, distance_m, camera_matrix),
            "wall_response": float(np.mean(responses)),
            "affine_params_wall_residual": residual_info["affine_params"],
            "prominent_threshold_local_residual_px": prominent_threshold,
            "distance_m": distance_m,
            "camera_params": camera_params,
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist()[0],
            "wall_roi": wall_roi,
            "ref_rois": ref_rois,
            **global_info,
        }

        annotated = draw_annotation(target_color_undist, wall_roi, ref_rois, cell_rows, summary)
        return jsonify(sanitize_for_json({
            "ok": True,
            "summary": summary,
            "cells": cell_rows,
            "annotated_png_base64": png_base64(annotated),
        }))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
