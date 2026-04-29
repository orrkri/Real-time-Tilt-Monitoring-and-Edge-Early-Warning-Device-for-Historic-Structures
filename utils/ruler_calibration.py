import cv2
import numpy as np


def calibrate_scale_from_ruler(ruler_img: np.ndarray, ruler_length_mm: float = 1000.0) -> float:
    """从 ruler 图标定像素到毫米的转换系数。

    方法：检测 ruler 的两条长边（垂直线），计算它们之间的像素距离。

    Args:
        ruler_img: ruler 彩色图
        ruler_length_mm: ruler 的物理长度，默认 1000mm（1m）

    Returns:
        k: 比例系数（mm/px）
    """
    gray = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=10)
    if lines is None or len(lines) < 2:
        raise RuntimeError("无法在 ruler 图中检测到足够的直线，请检查图像质量")

    # 筛选近似垂直的线（角度接近 90 度）
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 20:  # 近似垂直
            x_avg = (x1 + x2) / 2
            length = abs(y2 - y1)
            vertical_lines.append((x_avg, length))

    if len(vertical_lines) < 2:
        raise RuntimeError("无法找到两条垂直边来标定 ruler")

    # 按 x 坐标排序，取最左和最右的两条
    vertical_lines.sort(key=lambda t: t[0])
    left_x = vertical_lines[0][0]
    right_x = vertical_lines[-1][0]
    pixel_distance = abs(right_x - left_x)

    if pixel_distance < 10:
        raise RuntimeError("检测到的 ruler 边距过小，可能检测错误")

    k = ruler_length_mm / pixel_distance
    return k
