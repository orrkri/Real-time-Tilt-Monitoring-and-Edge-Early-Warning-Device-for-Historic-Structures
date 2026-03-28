%% ========== 相机标定 MATLAB 完整代码（修正版） ==========
clear; clc; close all;

%% 1. 设置参数
% 将路径改为与 Python 对应的路径
imageDir = 'calibration_images';   
squareSize = 3; % 棋盘格方格物理尺寸 (mm)

% 注：不需要像 Python 那样手动设置 11x8，MATLAB 会自动检测！

%% 2. 读取图片 + 检测角点
% 使用 imageDatastore 更高效地管理文件，支持多种后缀
imds = imageDatastore(imageDir, 'FileExtensions', {'.jpg', '.png', '.bmp'});

if isempty(imds.Files)
    error('未在 %s 找到任何图片，请检查路径和后缀是否正确！', imageDir);
end

fprintf('找到 %d 张图片，开始检测角点...\n', numel(imds.Files));

% 自动检测棋盘格角点
% 注意：boardSize 是返回值，不能作为输入传给此函数
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imds.Files);

validCount = sum(imagesUsed);
fprintf('成功提取角点的图片数量: %d / %d\n', validCount, numel(imds.Files));

if validCount == 0
    error('未能成功提取任何图片的角点，标定失败。请检查图片质量。');
end

% 获取图像分辨率（从第一张成功检测的图片中获取）
firstValidImg = readimage(imds, find(imagesUsed, 1));
imageSize = [size(firstValidImg, 1), size(firstValidImg, 2)];

% 生成世界坐标（Z=0平面）
% 这里的 boardSize 已经是 MATLAB 自动识别出的方格数（例如 [9, 12]）
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

%% 3. 相机标定（核心）
fprintf('正在进行相机标定，请稍候...\n');

% 开启切向畸变估计，使其与 OpenCV 的默认行为对齐
[params, ~, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'ImageSize', imageSize, ...
    'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 2);

%% 4. 显示标定结果
figure('Name', '重投影误差'); 
showReprojectionErrors(params);  
title('重投影误差（像素）');

figure('Name', '相机外参'); 
showExtrinsics(params);          
title('相机外参可视化');

%% 5. 输出标定参数（内参+畸变）
fprintf('\n========== 标定结果 ==========\n');
fprintf('图像分辨率: %d x %d\n', imageSize(2), imageSize(1));
fprintf('焦距 (fx, fy) = [%.6f, %.6f] 像素\n', params.FocalLength(1), params.FocalLength(2));
fprintf('主点 (cx, cy) = [%.6f, %.6f] 像素\n', params.PrincipalPoint(1), params.PrincipalPoint(2));
fprintf('径向畸变 k1 = %.6f, k2 = %.6f\n', params.RadialDistortion(1), params.RadialDistortion(2));
fprintf('切向畸变 p1 = %.6f, p2 = %.6f\n', params.TangentialDistortion(1), params.TangentialDistortion(2));
fprintf('平均重投影误差：%.6f 像素\n', mean(params.ReprojectionErrors(:)));

%% 6. 保存标定结果
save('cameraParams.mat', 'params', 'estimationErrors');
fprintf('\n标定完成，结果已保存至 cameraParams.mat\n');