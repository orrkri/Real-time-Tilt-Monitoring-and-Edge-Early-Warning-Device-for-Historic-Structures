import cv2
import numpy as np
import glob
import os
from pathlib import Path

# ================= 参数设置 =================
# 棋盘格内角点数量 (列数, 行数) -> 对应 12x9 个格子
CHECKERBOARD = (11, 8)
# 棋盘格单个方格的物理尺寸，单位为毫米 (mm)
SQUARE_SIZE = 3.0
# 存放用于标定的图片文件夹路径 (请替换为您自己的图片路径)
IMAGE_DIR = './calibration_images/*.jpg'
# 参与标定的最少有效图片数
MIN_VALID_IMAGES = 10
# 输出目录
OUTPUT_DIR = './calibration_output'
# 是否保存角点可视化图片
SAVE_CORNER_VIS = True
# ============================================

# 设置寻找亚像素角点的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备 3D 真实世界坐标点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_calibration_yaml(yaml_path: str, camera_matrix, dist_coeffs, image_size, rms, per_image_errors, used_images):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f'无法写入 YAML 文件: {yaml_path}')
    fs.write('image_width', int(image_size[0]))
    fs.write('image_height', int(image_size[1]))
    fs.write('checkerboard_cols', int(CHECKERBOARD[0]))
    fs.write('checkerboard_rows', int(CHECKERBOARD[1]))
    fs.write('square_size_mm', float(SQUARE_SIZE))
    fs.write('rms', float(rms))
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist_coeffs)
    fs.startWriteStruct('used_images', cv2.FileNode_SEQ)
    for img_path in used_images:
        fs.write('', str(img_path))
    fs.endWriteStruct()
    fs.startWriteStruct('per_image_errors', cv2.FileNode_SEQ)
    for err in per_image_errors:
        fs.write('', float(err))
    fs.endWriteStruct()
    fs.release()


def format_basename(path: str) -> str:
    return Path(path).name


# 用于存储所有图片的 3D 物体点和 2D 图像点
objpoints = []
imgpoints = []
used_image_paths = []
reference_image_path = None
reference_image = None
reference_image_size = None  # (width, height)
image_results = []
corner_output_dir = os.path.join(OUTPUT_DIR, 'corners')

images = sorted(glob.glob(IMAGE_DIR))

if not images:
    print(f'未在 {IMAGE_DIR} 找到任何图片，请检查路径！')
    raise SystemExit(1)

ensure_dir(OUTPUT_DIR)
if SAVE_CORNER_VIS:
    ensure_dir(corner_output_dir)

print(f'找到 {len(images)} 张图片，开始提取角点...')

for idx, fname in enumerate(images, start=1):
    basename = format_basename(fname)
    result = {
        'index': idx,
        'path': fname,
        'name': basename,
        'status': '',
        'reason': ''
    }

    img = cv2.imread(fname)
    if img is None:
        result['status'] = '失败'
        result['reason'] = '图像读取失败'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 失败 - 图像读取失败')
        continue

    height, width = img.shape[:2]
    current_size = (width, height)

    if reference_image_size is None:
        reference_image_size = current_size
        reference_image_path = fname
        reference_image = img.copy()
    elif current_size != reference_image_size:
        result['status'] = '跳过'
        result['reason'] = f'分辨率不一致，当前为 {current_size}，基准为 {reference_image_size}'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 跳过 - {result["reason"]}')
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if not found:
        result['status'] = '失败'
        result['reason'] = '未检测到完整棋盘格角点'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 失败 - 未检测到完整棋盘格角点')
        continue

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp.copy())
    imgpoints.append(corners2)
    used_image_paths.append(fname)

    result['status'] = '成功'
    result['reason'] = f'成功提取 {len(corners2)} 个角点'
    image_results.append(result)
    print(f'[{idx:02d}/{len(images):02d}] {basename}: 成功 - 成功提取 {len(corners2)} 个角点')

    if SAVE_CORNER_VIS:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)
        save_name = os.path.join(corner_output_dir, f'corners_{basename}')
        cv2.imwrite(save_name, vis)

valid_count = len(objpoints)
print('\n========= 角点提取统计 =========')
print(f'总图片数: {len(images)}')
print(f'成功用于标定: {valid_count}')
print(f'未成功/跳过: {len(images) - valid_count}')

if valid_count == 0:
    print('\n未能成功提取任何图片的角点，标定失败。请检查图片质量或检查角点参数 CHECKERBOARD 是否设置正确。')
    raise SystemExit(1)

if valid_count < MIN_VALID_IMAGES:
    print(f'\n警告：当前仅有 {valid_count} 张有效图片，低于建议的最少数量 {MIN_VALID_IMAGES} 张。')
    print('标定仍会继续，但结果稳定性可能不足。')

print('\n正在进行相机标定，请稍候...')
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, reference_image_size, None, None
)

print('\n========= 标定结果 =========')
print(f'标定使用图片数量: {valid_count} / {len(images)}')
print(f'图像分辨率: {reference_image_size[0]} x {reference_image_size[1]}')
print(f'OpenCV 返回的 RMS 重投影误差: {rms:.6f} 像素')
print('\n相机内参矩阵 (Camera Matrix):')
print(camera_matrix)
print('\n畸变系数 (Distortion Coefficients):')
print(dist_coeffs)
print(f'\nfx = {camera_matrix[0, 0]:.6f}')
print(f'fy = {camera_matrix[1, 1]:.6f}')
print(f'cx = {camera_matrix[0, 2]:.6f}')
print(f'cy = {camera_matrix[1, 2]:.6f}')

# 逐张计算重投影误差
per_image_errors = []
print('\n========= 每张图片的重投影误差 =========')
for i in range(valid_count):
    projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    per_image_errors.append(float(error))
    print(f'[{i + 1:02d}/{valid_count:02d}] {format_basename(used_image_paths[i])}: {error:.6f} 像素')

mean_error = float(np.mean(per_image_errors))
max_error = float(np.max(per_image_errors))
min_error = float(np.min(per_image_errors))
std_error = float(np.std(per_image_errors))

print('\n========= 重投影误差统计 =========')
print(f'总平均重投影误差: {mean_error:.6f} 像素 (越接近 0 越好)')
print(f'最小单张误差: {min_error:.6f} 像素')
print(f'最大单张误差: {max_error:.6f} 像素')
print(f'单张误差标准差: {std_error:.6f} 像素')

# 给出异常图片提示
threshold = max(mean_error + 2 * std_error, 0.15)
outliers = [(path, err) for path, err in zip(used_image_paths, per_image_errors) if err > threshold]
print('\n========= 异常图片检查 =========')
if outliers:
    print(f'检测到 {len(outliers)} 张可能的异常图片（阈值: {threshold:.6f} 像素）：')
    for path, err in outliers:
        print(f'- {format_basename(path)}: {err:.6f} 像素')
    print('建议优先检查这些图片是否模糊、反光、姿态极端，必要时剔除后重新标定。')
else:
    print(f'未发现明显异常图片（判定阈值: {threshold:.6f} 像素）。')

# 保存标定结果
npz_path = os.path.join(OUTPUT_DIR, 'calibration_result.npz')
np.savez(
    npz_path,
    checkerboard=np.array(CHECKERBOARD, dtype=np.int32),
    square_size_mm=np.array([SQUARE_SIZE], dtype=np.float32),
    image_size=np.array(reference_image_size, dtype=np.int32),
    rms=np.array([rms], dtype=np.float64),
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=np.array(rvecs, dtype=object),
    tvecs=np.array(tvecs, dtype=object),
    used_images=np.array(used_image_paths, dtype=object),
    per_image_errors=np.array(per_image_errors, dtype=np.float64),
)

yaml_path = os.path.join(OUTPUT_DIR, 'calibration_result.yaml')
save_calibration_yaml(
    yaml_path,
    camera_matrix,
    dist_coeffs,
    reference_image_size,
    rms,
    per_image_errors,
    used_image_paths,
)

summary_path = os.path.join(OUTPUT_DIR, 'calibration_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('========= 相机标定汇总 =========\n')
    f.write(f'图片匹配模式: {IMAGE_DIR}\n')
    f.write(f'总图片数: {len(images)}\n')
    f.write(f'成功用于标定: {valid_count}\n')
    f.write(f'图像分辨率: {reference_image_size[0]} x {reference_image_size[1]}\n')
    f.write(f'棋盘格内角点: {CHECKERBOARD}\n')
    f.write(f'方格尺寸(mm): {SQUARE_SIZE}\n')
    f.write(f'RMS 重投影误差: {rms:.6f}\n')
    f.write(f'总平均重投影误差: {mean_error:.6f}\n')
    f.write(f'最小单张误差: {min_error:.6f}\n')
    f.write(f'最大单张误差: {max_error:.6f}\n')
    f.write(f'单张误差标准差: {std_error:.6f}\n\n')
    f.write('相机内参矩阵:\n')
    f.write(np.array2string(camera_matrix, precision=8))
    f.write('\n\n畸变系数:\n')
    f.write(np.array2string(dist_coeffs, precision=8))
    f.write('\n\n每张图片结果:\n')
    for result in image_results:
        f.write(f"{result['name']}: {result['status']} - {result['reason']}\n")
    f.write('\n每张已参与标定图片的重投影误差:\n')
    for path, err in zip(used_image_paths, per_image_errors):
        f.write(f'{format_basename(path)}: {err:.6f}\n')

# 生成去畸变验证图
if reference_image is not None:
    h, w = reference_image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(reference_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted_cropped = undistorted[y:y + rh, x:x + rw]
    else:
        undistorted_cropped = undistorted

    raw_save_path = os.path.join(OUTPUT_DIR, f'undistort_source_{format_basename(reference_image_path)}')
    undistort_save_path = os.path.join(OUTPUT_DIR, f'undistorted_{format_basename(reference_image_path)}')
    undistort_crop_save_path = os.path.join(OUTPUT_DIR, f'undistorted_cropped_{format_basename(reference_image_path)}')
    cv2.imwrite(raw_save_path, reference_image)
    cv2.imwrite(undistort_save_path, undistorted)
    cv2.imwrite(undistort_crop_save_path, undistorted_cropped)

    print('\n========= 去畸变验证图 =========')
    print(f'原始验证图已保存: {raw_save_path}')
    print(f'去畸变图已保存: {undistort_save_path}')
    print(f'裁剪后去畸变图已保存: {undistort_crop_save_path}')

print('\n========= 文件输出 =========')
print(f'参数 npz 文件: {npz_path}')
print(f'参数 yaml 文件: {yaml_path}')
print(f'汇总文本文件: {summary_path}')
if SAVE_CORNER_VIS:
    print(f'角点可视化目录: {corner_output_dir}')

print('\n标定完成。')
