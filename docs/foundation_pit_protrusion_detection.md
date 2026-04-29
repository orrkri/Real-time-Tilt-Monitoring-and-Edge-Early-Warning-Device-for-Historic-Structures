# 基坑侧面凸起墙面检测算法设计

## 1. 问题定义

目标：从固定相机拍摄的基坑侧面图像中，自动检测并量化墙面的局部凸起（protrusion / bulge）。

## 2. 与木棒延伸检测的异同

| 维度 | 木棒延伸检测 | 基坑墙面凸起检测 |
|------|-------------|-----------------|
| 运动类型 | 刚体平移 | 局部形变 / 鼓出 |
| 先验知识 | 有 0mm 基线 | 有"平整期"基线 |
| 特征 | 长条状、高对比度 | 大面积、低对比度、纹理重复 |
| 尺度 | 5-20 mm | 可能 10-100+ mm |
| 方向 | 主要是水平位移 | 主要是法向（朝向相机）位移 |

## 3. 核心难点

1. **法向位移在图像上的投影小**：若相机近似正对墙面，凸起主要表现为微弱的局部放大（纹理变粗）和阴影变化，横向位移不明显。
2. **光照与天气**：基坑环境光照变化剧烈，不能依赖简单的背景差分阈值。
3. **纹理重复**：土体、混凝土支护的纹理高度重复，传统角点匹配容易误匹配。
4. **无纹理区域**：部分墙面可能是光滑混凝土或喷浆面，缺乏可跟踪特征。

## 4. 推荐算法路线

综合考虑现有标定体系、单目相机约束、实时性需求，推荐 **"轮廓偏差法 + 光流膨胀法"双通道** 架构。

### 4.1 通道 A：轮廓偏差法（侧视相机适用）

适用场景：相机位于基坑边缘，斜向俯视/侧视坑壁，能看到墙面的纵向轮廓。

**步骤：**

1. **图像配准（Image Registration）**
   - 输入：平整期基线图像 $I_0$、当前图像 $I_t$
   - 方法：相位相关（phaseCorrelate）或基于 Harris/SIFT 特征的单应性矩阵 $H$
   - 输出：对齐后的图像 $I_t'$

2. **ROI 与轮廓提取**
   - 在 $I_0$ 中人工或自动标定墙面纵向轮廓区域（多边形 ROI）
   - 对 $I_t'$ 在 ROI 内逐行（或每隔 N 行）检测墙面边缘
   - 边缘检测：使用 Canny + 垂直梯度加权，或直接用水平 Sobel 的局部极值

3. **轮廓点云构建**
   - 收集所有行检测到的边缘点 $(x_i, y_i)$，构成二维轮廓 $P$
   - 对 $P$ 做中值滤波或 Savitzky-Golay 平滑，去除孤立噪声点

4. **基线拟合**
   - 在 $I_0$ 上同样提取轮廓 $P_0$
   - 对 $P_0$ 拟合参考曲线（直线、二次曲线或 B-spline）$C_{ref}(y)$

5. **偏差计算**
   - 对 $P$ 中每个点计算到 $C_{ref}$ 的有向距离 $d_i$
   - 1面场景（轮廓向左为凸起）：$d_i = x_{ref}(y_i) - x_i$
   - 2面场景（轮廓向右为凸起）：$d_i = x_i - x_{ref}(y_i)$

6. **凸起判定**
   - 对 $d_i$ 做一维信号处理：
     - 找连通区域：连续 $N$ 行以上 $d_i > T_{pix}$
     - 或求局部极大值：$d_i$ 的峰值超过 $T_{pix}$
   - 将像素偏差转换为物理尺寸：$D_{mm} = d_i \cdot k$，其中 $k$ 为当前距离下的 mm/px 比例系数

**阈值建议（需根据现场标定调整）：**
- $T_{pix}$：2-5 px（取决于相机分辨率）
- 连续行数 $N$：≥ 10 行（过滤噪声）

### 4.2 通道 B：光流膨胀法（正视/弱侧视相机适用）

适用场景：相机大致正对墙面，轮廓不明显，但凸起会导致局部纹理产生径向膨胀（朝向凸起的中心，纹理看起来被轻微放大）。

**步骤：**

1. **图像分块**
   - 将 ROI 划分为 $M \times N$ 的网格（如 32×32 px 的 cell）

2. **稠密光流计算**
   - 使用 Farneback 光流：$\text{cv2.calcOpticalFlowFarneback}(I_0, I_t', ...)$
   - 得到每个像素的光流向量 $(u, v)$

3. **膨胀特征提取**
   - 计算光流散度（divergence）：$\nabla \cdot \mathbf{F} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$
   - 凸起区域在正视图下会产生正的散度（纹理向外膨胀）
   - 对每个 cell 计算平均散度 $\bar{\delta}$

4. **异常检测**
   - 设定散度阈值 $T_{div} > 0$
   - 将 $|\bar{\delta}| > T_{div}$ 且空间连通的 cell 标记为候选凸起区域

5. **物理尺寸估计**
   - 若相机近似正视，法向位移 $\Delta Z$ 与局部放大率 $m$ 的关系近似为：$\Delta Z \approx Z \cdot (m - 1)$
   - 放大率可通过光流散度积分估算，或直接用像素面积变化比例近似

### 4.3 双通道融合策略

```
if 相机侧视角度 > 30°:
    以轮廓偏差法为主
    光流法作为辅助验证
else:
    以光流膨胀法为主
    轮廓法作为辅助（若轮廓可见）

最终输出：两个通道同时报警才判定为高风险凸起（降低误报）
```

## 5. 原型伪代码

```python
import cv2
import numpy as np
from scipy.signal import savgol_filter

def detect_wall_protrusion(
    img_curr: np.ndarray,
    img_baseline: np.ndarray,
    face: str,
    roi_polygon: np.ndarray,   # (N, 2) 多边形顶点
    distance_m: float,
    fx: float,
    pix_threshold: float = 3.0,
    min_continuous_rows: int = 10,
):
    h, w = img_curr.shape[:2]
    k = (distance_m * 1000.0) / fx  # mm/px

    # 1. 配准
    gray0 = cv2.cvtColor(img_baseline, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray1 = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (sx, sy), _ = cv2.phaseCorrelate(gray0, gray1)
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    gray1_aligned = cv2.warpAffine(gray1, M, (w, h))

    # 2. ROI mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)

    # 3. 逐行提取轮廓
    profile_x = []
    profile_y = []
    for y in range(h):
        row_mask = mask[y, :]
        if not np.any(row_mask):
            continue
        xs = np.where(row_mask > 0)[0]
        x_start, x_end = xs.min(), xs.max()
        row = gray1_aligned[y, x_start:x_end]
        # 水平梯度极值 = 墙面边缘
        gx = cv2.Sobel(row, cv2.CV_64F, 1, 0, ksize=3).flatten()
        edge_local = np.argmax(np.abs(gx))
        edge_x = x_start + edge_local
        profile_x.append(edge_x)
        profile_y.append(y)

    profile_x = np.array(profile_x, dtype=np.float32)
    profile_y = np.array(profile_y, dtype=np.float32)

    # 4. 平滑
    if len(profile_x) > 21:
        profile_x_smooth = savgol_filter(profile_x, window_length=21, polyorder=3)
    else:
        profile_x_smooth = profile_x

    # 5. 基线（从 baseline 同样提取）
    # 实际工程中可离线计算一次并缓存
    baseline_x = ...  # 同样流程提取

    # 6. 偏差
    if face == "1面":
        deviation_px = baseline_x - profile_x_smooth
    else:
        deviation_px = profile_x_smooth - baseline_x

    # 7. 凸起判定
    protrusions = []
    in_protrusion = False
    start_y = 0
    for i, d in enumerate(deviation_px):
        if d > pix_threshold:
            if not in_protrusion:
                in_protrusion = True
                start_y = profile_y[i]
            max_d = d
        else:
            if in_protrusion:
                end_y = profile_y[i]
                height_px = end_y - start_y
                if height_px >= min_continuous_rows:
                    protrusions.append({
                        "y_range": (start_y, end_y),
                        "max_deviation_px": max_d,
                        "max_deviation_mm": max_d * k,
                    })
            in_protrusion = False

    return protrusions, deviation_px
```

## 6. 关键参数标定建议

1. **比例系数 $k$**：与木棒检测一致，使用 $k = \frac{Z \cdot 1000}{f_x}$，其中 $Z$ 为相机到墙面的垂直距离。
2. **轮廓提取方向**：若墙面是竖直的，应在每行（水平方向）搜索边缘；若相机视角导致墙面近似水平线，则改为逐列搜索。
3. **ROI 标定**：首次部署时需在基线图像上绘制墙面多边形 ROI，建议保存为 JSON/YAML 供后续自动加载。

## 7. 局限性与改进方向

| 局限 | 改进方向 |
|------|---------|
| 单目无法直接测深度 | 引入双目 stereo 或激光线结构光 |
| 光照剧烈变化导致轮廓漂移 | 增加 HDR 融合或使用红外补光 |
| 小凸起（< 2 px）不可见 | 提高相机分辨率或减小拍摄距离 |
| 工人/设备遮挡 | 增加目标检测（YOLO）剔除遮挡区域后再做轮廓分析 |
| 坑壁喷浆后无纹理 | 在墙面粘贴随机散斑或荧光标记点 |

## 8. 与现有代码的复用点

- `utils.calibration.load_calibration` / `undistort_image`：直接复用
- `detect_rod_extension.py` 中的 `imread_safe`：直接复用
- `parse_filename`：可扩展为解析基坑监测图像命名
- 相位相关配准逻辑：从 `utils.rod_detector` 迁移
- `evaluate_error.py`：评估框架可直接用于评估凸起检测误差
