import cv2
import numpy as np
import glob
import os

# ================= 参数设置 =================
# 棋盘格内角点数量 (列数, 行数) -> 对应 12x9 个格子
CHECKERBOARD = (11, 8) 
# 棋盘格单个方格的物理尺寸，单位为毫米 (mm)
SQUARE_SIZE = 3.0      
# 存放用于标定的图片文件夹路径 (请替换为您自己的图片路径)
IMAGE_DIR = './calibration_images/*.jpg' 
# ============================================

# 设置寻找亚像素角点的终止条件
# maxCount=30 (最大迭代次数), epsilon=0.001 (角点位置精度)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备 3D 真实世界坐标点
# 形式为 (0,0,0), (1,0,0), (2,0,0) ... (10,7,0)
# 然后乘以 SQUARE_SIZE，得到实际的毫米坐标
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# 用于存储所有图片的 3D 物体点和 2D 图像点
objpoints = [] # 真实世界中的 3D 点
imgpoints = [] # 图像中的 2D 点

# 获取所有标定图像
images = glob.glob(IMAGE_DIR)

if not images:
    print(f"未在 {IMAGE_DIR} 找到任何图片，请检查路径！")
else:
    print(f"找到 {len(images)} 张图片，开始提取角点...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 如果找到足够的角点
    if ret == True:
        objpoints.append(objp)

        # 进一步提取亚像素角点，提高标定精度
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 在图像上绘制并显示角点 (可选，用于调试)
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# 如果成功提取到角点，开始进行相机标定
if len(objpoints) > 0:
    print("\n正在进行相机标定，请稍候...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n========= 标定结果 =========")
    print("相机内参矩阵 (Camera Matrix):")
    print(mtx)
    print("\n畸变系数 (Distortion Coefficients):")
    print(dist)
    
    # 计算重投影误差，评估标定结果的准确性
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\n总平均重投影误差: {mean_error/len(objpoints):.4f} 像素 (越接近0越好)")
else:
    print("\n未能成功提取任何图片的角点，标定失败。请检查图片质量或检查角点参数 CHECKERBOARD 是否设置正确。")