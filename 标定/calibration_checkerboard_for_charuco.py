import cv2
import numpy as np
import glob
import os
from pathlib import Path

# ================= 参数设置 =================
# 这块板按“黑白格”理解：图案 660x480 mm，单格 60 mm => 11x8 个方格
# 因此棋盘格内角点数量为 (11-1, 8-1) = (10, 7)
CHECKERBOARD = (10, 7)   # (cols, rows) inner corners
SQUARE_SIZE = 60.0       # mm

# 自动读取多种格式
IMAGE_DIR = './calibration_images'
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

MIN_VALID_IMAGES = 8
OUTPUT_DIR = './calibration_output_checkerboard'
SAVE_CORNER_VIS = True
UPSCALE_FACTORS = [1.0, 1.5, 2.0, 3.0]
# ============================================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str, exts):
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(images)


def format_basename(path: str) -> str:
    return Path(path).name


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_variants(gray: np.ndarray):
    variants = []
    h, w = gray.shape[:2]

    def add_variant(name: str, img: np.ndarray, scale: float):
        variants.append((name, img, scale))

    for scale in UPSCALE_FACTORS:
        if scale == 1.0:
            base = gray.copy()
        else:
            base = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        add_variant(f'gray_x{scale:g}', base, scale)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(base)
        add_variant(f'clahe_x{scale:g}', clahe, scale)

        # sharpen
        blur = cv2.GaussianBlur(base, (0, 0), 1.0)
        sharp = cv2.addWeighted(base, 1.6, blur, -0.6, 0)
        add_variant(f'sharp_x{scale:g}', sharp, scale)

        # binary + close: 关键步骤，把白格中的 ArUco 小黑块尽量“填平”为白格
        bin_img = cv2.adaptiveThreshold(
            base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
        )
        # 核尺寸跟随图像大小变化，远距离小板时先放大再闭运算
        k = max(5, int(round(min(base.shape[:2]) / 120)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        add_variant(f'binary_close_x{scale:g}', closed, scale)

        # 更激进一点的闭运算版本
        k2 = max(k + 4, 9)
        if k2 % 2 == 0:
            k2 += 1
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
        closed2 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel2, iterations=2)
        add_variant(f'binary_close_strong_x{scale:g}', closed2, scale)

    return variants


def detect_checkerboard(gray: np.ndarray):
    # 先尝试更稳的 SB 检测器
    flags_sb = (
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_EXHAUSTIVE |
        cv2.CALIB_CB_ACCURACY
    )

    variants = make_variants(gray)
    best = None

    for name, img, scale in variants:
        try:
            found, corners = cv2.findChessboardCornersSB(img, CHECKERBOARD, flags=flags_sb)
        except Exception:
            found, corners = False, None

        if found and corners is not None and len(corners) == CHECKERBOARD[0] * CHECKERBOARD[1]:
            if scale != 1.0:
                corners = corners / scale
            score = len(corners)
            best = (name, corners, score)
            break

    if best is not None:
        name, corners, score = best
        corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (7, 7), (-1, -1), criteria)
        return True, corners, name

    # 再尝试传统 findChessboardCorners，配合闭运算图
    flags_std = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for name, img, scale in variants:
        found, corners = cv2.findChessboardCorners(img, CHECKERBOARD, flags_std)
        if found and corners is not None and len(corners) == CHECKERBOARD[0] * CHECKERBOARD[1]:
            if scale != 1.0:
                corners = corners / scale
            corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
            return True, corners, name

    return False, None, ''


def save_calibration_yaml(yaml_path: str, camera_matrix, dist_coeffs, image_size, rms, per_image_errors, used_images):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f'无法写入 YAML 文件: {yaml_path}')
    fs.write('image_width', int(image_size[0]))
    fs.write('image_height', int(image_size[1]))
    fs.write('pattern_type', 'checkerboard_from_charuco_board')
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


objpoints = []
imgpoints = []
used_image_paths = []
reference_image_path = None
reference_image = None
reference_image_size = None
image_results = []
corner_output_dir = os.path.join(OUTPUT_DIR, 'corners')

images = list_images(IMAGE_DIR, IMAGE_EXTS)
if not images:
    print(f'未在 {IMAGE_DIR} 找到图片，请检查路径。')
    raise SystemExit(1)

ensure_dir(OUTPUT_DIR)
if SAVE_CORNER_VIS:
    ensure_dir(corner_output_dir)

print(f'找到 {len(images)} 张图片，开始提取黑白格角点...')
print(f'目标模式：{CHECKERBOARD[0]} x {CHECKERBOARD[1]} 个内角点，方格尺寸 {SQUARE_SIZE} mm')

for idx, fname in enumerate(images, start=1):
    basename = format_basename(fname)
    result = {'index': idx, 'path': fname, 'name': basename, 'status': '', 'reason': ''}

    img = cv2.imread(fname)
    if img is None:
        result['status'] = '失败'
        result['reason'] = '图像读取失败'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 失败 - 图像读取失败')
        continue

    h, w = img.shape[:2]
    current_size = (w, h)
    if reference_image_size is None:
        reference_image_size = current_size
        reference_image_path = fname
        reference_image = img.copy()
    elif current_size != reference_image_size:
        result['status'] = '跳过'
        result['reason'] = f'分辨率不一致，当前 {current_size}，基准 {reference_image_size}'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 跳过 - {result["reason"]}')
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = variance_of_laplacian(gray)
    found, corners, variant_name = detect_checkerboard(gray)

    if not found:
        result['status'] = '失败'
        result['reason'] = f'未检测到完整棋盘格内角点，清晰度={sharpness:.2f}'
        image_results.append(result)
        print(f'[{idx:02d}/{len(images):02d}] {basename}: 失败 - {result["reason"]}')
        continue

    objpoints.append(objp.copy())
    imgpoints.append(corners)
    used_image_paths.append(fname)
    result['status'] = '成功'
    result['reason'] = f'成功提取 {len(corners)} 个角点，清晰度={sharpness:.2f}，方案={variant_name}'
    image_results.append(result)
    print(f'[{idx:02d}/{len(images):02d}] {basename}: 成功 - {result["reason"]}')

    if SAVE_CORNER_VIS:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, True)
        cv2.imwrite(os.path.join(corner_output_dir, f'corners_{basename}'), vis)

valid_count = len(objpoints)
print('\n========= 角点提取统计 =========')
print(f'总图片数: {len(images)}')
print(f'成功用于标定: {valid_count}')
print(f'未成功/跳过: {len(images) - valid_count}')

if valid_count == 0:
    print('未检测到任何有效图片，标定终止。')
    raise SystemExit(1)

if valid_count < MIN_VALID_IMAGES:
    print(f'警告：当前仅有 {valid_count} 张有效图片，低于建议数量 {MIN_VALID_IMAGES}。')

print('\n正在进行相机标定...')
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, reference_image_size, None, None
)

print('\n========= 标定结果 =========')
print(f'标定使用图片数量: {valid_count} / {len(images)}')
print(f'图像分辨率: {reference_image_size[0]} x {reference_image_size[1]}')
print(f'RMS 重投影误差: {rms:.6f} 像素')
print('\n相机内参矩阵:')
print(camera_matrix)
print('\n畸变系数:')
print(dist_coeffs)

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

threshold = max(mean_error + 2 * std_error, 0.20)
outliers = [(p, e) for p, e in zip(used_image_paths, per_image_errors) if e > threshold]

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
save_calibration_yaml(yaml_path, camera_matrix, dist_coeffs, reference_image_size, rms, per_image_errors, used_image_paths)

summary_path = os.path.join(OUTPUT_DIR, 'calibration_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('========= 相机标定汇总 =========\n')
    f.write('模式说明：将当前 ChArUco 板按黑白棋盘格处理\n')
    f.write(f'图片目录: {IMAGE_DIR}\n')
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
    f.write('\n异常图片建议检查:\n')
    if outliers:
        for path, err in outliers:
            f.write(f'- {format_basename(path)}: {err:.6f}\n')
    else:
        f.write('无明显异常图片\n')

if reference_image is not None:
    h, w = reference_image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(reference_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted_cropped = undistorted[y:y + rh, x:x + rw]
    else:
        undistorted_cropped = undistorted

    cv2.imwrite(os.path.join(OUTPUT_DIR, f'undistort_source_{format_basename(reference_image_path)}'), reference_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'undistorted_{format_basename(reference_image_path)}'), undistorted)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'undistorted_cropped_{format_basename(reference_image_path)}'), undistorted_cropped)

print('\n========= 重投影误差统计 =========')
print(f'总平均重投影误差: {mean_error:.6f} 像素')
print(f'最小单张误差: {min_error:.6f} 像素')
print(f'最大单张误差: {max_error:.6f} 像素')
print(f'单张误差标准差: {std_error:.6f} 像素')

print('\n========= 异常图片检查 =========')
if outliers:
    print(f'检测到 {len(outliers)} 张可能的异常图片（阈值: {threshold:.6f} 像素）:')
    for path, err in outliers:
        print(f'- {format_basename(path)}: {err:.6f} 像素')
else:
    print(f'未发现明显异常图片（阈值: {threshold:.6f} 像素）。')

print('\n========= 文件输出 =========')
print(f'参数 npz 文件: {npz_path}')
print(f'参数 yaml 文件: {yaml_path}')
print(f'汇总文本文件: {summary_path}')
if SAVE_CORNER_VIS:
    print(f'角点可视化目录: {corner_output_dir}')

print('\n标定完成。')
