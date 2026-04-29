import cv2
import numpy as np


def detect_rod_extension_pixels(
    img: np.ndarray,
    baseline_img: np.ndarray,
    face: str,
    max_search_px: int = 20,
    corr_threshold: float = 0.90,
) -> tuple:
    """检测木棒延伸的像素量（基于 1D 互相关）。

    策略：将测试图与 0mm 基线图在 ROI 内做垂直投影，
    通过一维互相关计算水平位移。

    Args:
        img: 待测图（已去畸变）
        baseline_img: 0mm 基线图（已去畸变，同分辨率）
        face: "1面" 或 "2面"
        max_search_px: 最大搜索像素位移
        corr_threshold: 互相关最低阈值，低于此值认为检测失败

    Returns:
        (extension_pixels, tip_x, debug_mask)
        extension_pixels: 木棒端点到平台边缘的像素距离（正值）
        tip_x: 木棒端点的 x 坐标（全图坐标）
        debug_mask: ROI 掩码（调试用）
    """
    if img.shape != baseline_img.shape:
        raise ValueError(f"图像分辨率不一致: {img.shape} vs {baseline_img.shape}")

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_base = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if face == "1面":
        roi_y_end = int(h * 0.6)
        roi_x_end = int(w * 0.6)
        roi = gray[:roi_y_end, :roi_x_end]
        roi_base = gray_base[:roi_y_end, :roi_x_end]
    elif face == "2面":
        roi_y_end = int(h * 0.6)
        roi_x_start = int(w * 0.4)
        roi = gray[:roi_y_end, roi_x_start:]
        roi_base = gray_base[:roi_y_end, roi_x_start:]
    else:
        raise ValueError(f"未知的 face: {face}，应为 '1面' 或 '2面'")

    profile = np.sum(roi, axis=0)
    profile_base = np.sum(roi_base, axis=0)

    best_lag = 0
    best_r = -1.0
    for lag in range(-max_search_px, max_search_px + 1):
        if lag < 0:
            a = profile_base[:lag]
            b = profile[-lag:]
        elif lag > 0:
            a = profile_base[lag:]
            b = profile[:-lag]
        else:
            a = profile_base
            b = profile
        if len(a) > 10:
            r = np.corrcoef(a, b)[0, 1]
            if not np.isnan(r) and r > best_r:
                best_r = r
                best_lag = lag

    if best_r < corr_threshold:
        raise RuntimeError(
            f"相关性过低 ({best_r:.3f} < {corr_threshold})，无法检测位移"
        )

    baseline_x = _detect_platform_edge(baseline_img, face)

    if face == "1面":
        extension_pixels = max(0, best_lag)
        tip_x = baseline_x - extension_pixels
    else:
        extension_pixels = max(0, -best_lag)
        tip_x = baseline_x + extension_pixels

    debug_mask = np.zeros((h, w), dtype=np.uint8)
    if face == "1面":
        debug_mask[:roi_y_end, :roi_x_end] = 128
    else:
        debug_mask[:roi_y_end, roi_x_start:] = 128

    return extension_pixels, int(tip_x), debug_mask


def _detect_platform_edge(empty_img: np.ndarray, face: str, roi_ratio: float = 0.3) -> int:
    """从 empty 图中检测平台边缘的 x 坐标。"""
    gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if face == "1面":
        roi = gray[:, :int(w * roi_ratio)]
        edges = cv2.Canny(roi, 50, 150)
        x_profile = np.sum(edges, axis=0)
        edge_x = int(np.argmax(x_profile))
        return edge_x
    else:
        roi = gray[:, int(w * (1 - roi_ratio)):]
        edges = cv2.Canny(roi, 50, 150)
        x_profile = np.sum(edges, axis=0)
        edge_x = int(np.argmax(x_profile)) + int(w * (1 - roi_ratio))
        return edge_x
