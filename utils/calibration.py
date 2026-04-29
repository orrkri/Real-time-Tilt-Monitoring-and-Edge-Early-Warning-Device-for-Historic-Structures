import cv2
import numpy as np
from pathlib import Path


def load_calibration(yaml_path: str):
    """读取 OpenCV FileStorage YAML 标定结果。

    Returns:
        camera_matrix (np.ndarray): 3x3 内参矩阵
        dist_coeffs (np.ndarray): 1x5 畸变系数 [k1, k2, p1, p2, k3]
        image_size (tuple): (width, height)
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开标定文件: {yaml_path}")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    fs.release()

    return camera_matrix, dist_coeffs, (width, height)


def undistort_image(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """对输入图像进行去畸变。"""
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
    return undistorted
