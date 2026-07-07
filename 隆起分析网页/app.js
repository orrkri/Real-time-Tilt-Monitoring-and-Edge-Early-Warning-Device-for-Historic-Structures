const DEFAULT_REF_ROI_25M = { x0: 20, y0: 80, x1: 330, y1: 520 };
const DEFAULT_WALL_ROI_25M = { x0: 601, y0: 964, x1: 1115, y1: 1607 };
const DEFAULT_CAMERA = {
  fx: 4900.35287,
  fy: 4897.45097,
  cx: 1188.68904,
  cy: 1033.54513,
  k1: -0.0998273084,
  k2: -0.311407213,
  p1: -0.00021817366,
  p2: -0.000158409673,
  k3: 1.70263913,
};

const els = {
  baseInput: document.getElementById("baseInput"),
  targetInput: document.getElementById("targetInput"),
  colsInput: document.getElementById("colsInput"),
  rowsInput: document.getElementById("rowsInput"),
  distanceInput: document.getElementById("distanceInput"),
  fxInput: document.getElementById("fxInput"),
  fyInput: document.getElementById("fyInput"),
  cxInput: document.getElementById("cxInput"),
  cyInput: document.getElementById("cyInput"),
  k1Input: document.getElementById("k1Input"),
  k2Input: document.getElementById("k2Input"),
  p1Input: document.getElementById("p1Input"),
  p2Input: document.getElementById("p2Input"),
  k3Input: document.getElementById("k3Input"),
  resetCameraBtn: document.getElementById("resetCameraBtn"),
  refModeBtn: document.getElementById("refModeBtn"),
  wallModeBtn: document.getElementById("wallModeBtn"),
  useDefaultRoi: document.getElementById("useDefaultRoi"),
  clearRoi: document.getElementById("clearRoi"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  baseTab: document.getElementById("baseTab"),
  targetTab: document.getElementById("targetTab"),
  statusText: document.getElementById("statusText"),
  canvas: document.getElementById("viewCanvas"),
  summaryPanel: document.getElementById("summaryPanel"),
  predictedBulgeValue: document.getElementById("predictedBulgeValue"),
  medianValue: document.getElementById("medianValue"),
  residualP95Value: document.getElementById("residualP95Value"),
  stdValue: document.getElementById("stdValue"),
  globalCcValue: document.getElementById("globalCcValue"),
  qualityValue: document.getElementById("qualityValue"),
  downloadPng: document.getElementById("downloadPng"),
  downloadCsv: document.getElementById("downloadCsv"),
  tableWrap: document.getElementById("tableWrap"),
  detailRows: document.getElementById("detailRows"),
};

const ctx = els.canvas.getContext("2d");

const state = {
  base: null,
  target: null,
  refRois: [],
  wallRoi: null,
  drawMode: "ref",
  dragStart: null,
  view: "base",
  result: null,
};

function setStatus(text) {
  els.statusText.value = text;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fmt(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function normalizeRect(a, b) {
  return {
    x0: Math.min(a.x, b.x),
    y0: Math.min(a.y, b.y),
    x1: Math.max(a.x, b.x),
    y1: Math.max(a.y, b.y),
  };
}

function rectWidth(rect) {
  return rect.x1 - rect.x0;
}

function rectHeight(rect) {
  return rect.y1 - rect.y0;
}

function getCanvasPoint(evt) {
  const rect = els.canvas.getBoundingClientRect();
  const sx = els.canvas.width / rect.width;
  const sy = els.canvas.height / rect.height;
  return {
    x: clamp((evt.clientX - rect.left) * sx, 0, els.canvas.width - 1),
    y: clamp((evt.clientY - rect.top) * sy, 0, els.canvas.height - 1),
  };
}

function drawBaseImage(image) {
  els.canvas.width = image.width;
  els.canvas.height = image.height;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.drawImage(image, 0, 0);
}

function drawRect(rect, color, fill, label, width = 4) {
  if (!rect) return;
  ctx.save();
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.setLineDash([12, 8]);
  ctx.strokeRect(rect.x0, rect.y0, rectWidth(rect), rectHeight(rect));
  ctx.setLineDash([]);
  ctx.fillStyle = fill;
  ctx.fillRect(rect.x0, rect.y0, rectWidth(rect), rectHeight(rect));
  ctx.font = "18px Microsoft YaHei, Arial";
  ctx.fillStyle = color;
  ctx.fillText(label, rect.x0 + 8, Math.max(24, rect.y0 - 10));
  ctx.restore();
}

function drawSelections() {
  state.refRois.forEach((roi, idx) => {
    drawRect(roi, "#2563eb", "rgba(37, 99, 235, 0.12)", `REF_${idx + 1}`);
  });
  drawRect(state.wallRoi, "#ef4444", "rgba(239, 68, 68, 0.12)", "WALL");

  if (state.dragStart) {
    const color = state.drawMode === "ref" ? "#2563eb" : "#ef4444";
    const fill = state.drawMode === "ref" ? "rgba(37, 99, 235, 0.14)" : "rgba(239, 68, 68, 0.14)";
    drawRect(normalizeRect(state.dragStart, state.dragStart.current), color, fill, "DRAW", 3);
  }
}

function render() {
  if (state.view === "target" && state.result?.annotatedImage) {
    drawBaseImage(state.result.annotatedImage);
    return;
  }

  const image = state.view === "target" && state.target ? state.target.image : state.base?.image;
  if (!image) {
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    return;
  }

  drawBaseImage(image);
  drawSelections();
}

async function imageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片读取失败"));
    img.src = url;
  });
}

async function loadImageFile(file) {
  const url = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  const image = await imageFromUrl(url);
  return { image, file, name: file.name, width: image.width, height: image.height };
}

function setDrawMode(mode) {
  state.drawMode = mode;
  els.refModeBtn.classList.toggle("active", mode === "ref");
  els.wallModeBtn.classList.toggle("active", mode === "wall");
  setStatus(mode === "ref" ? "当前模式：框固定参考区。可连续框多个蓝色参考区。" : "当前模式：框墙体分析区。红色区域只保留最后一次框选。");
}

function readNumber(input, fallback, label) {
  const value = Number.parseFloat(input.value);
  if (!Number.isFinite(value)) {
    throw new Error(`${label} 不是有效数字`);
  }
  return value;
}

function readCameraParams() {
  return {
    fx: readNumber(els.fxInput, DEFAULT_CAMERA.fx, "fx"),
    fy: readNumber(els.fyInput, DEFAULT_CAMERA.fy, "fy"),
    cx: readNumber(els.cxInput, DEFAULT_CAMERA.cx, "cx"),
    cy: readNumber(els.cyInput, DEFAULT_CAMERA.cy, "cy"),
    k1: readNumber(els.k1Input, DEFAULT_CAMERA.k1, "k1"),
    k2: readNumber(els.k2Input, DEFAULT_CAMERA.k2, "k2"),
    p1: readNumber(els.p1Input, DEFAULT_CAMERA.p1, "p1"),
    p2: readNumber(els.p2Input, DEFAULT_CAMERA.p2, "p2"),
    k3: readNumber(els.k3Input, DEFAULT_CAMERA.k3, "k3"),
  };
}

function resetCameraParams() {
  els.fxInput.value = DEFAULT_CAMERA.fx;
  els.fyInput.value = DEFAULT_CAMERA.fy;
  els.cxInput.value = DEFAULT_CAMERA.cx;
  els.cyInput.value = DEFAULT_CAMERA.cy;
  els.k1Input.value = DEFAULT_CAMERA.k1;
  els.k2Input.value = DEFAULT_CAMERA.k2;
  els.p1Input.value = DEFAULT_CAMERA.p1;
  els.p2Input.value = DEFAULT_CAMERA.p2;
  els.k3Input.value = DEFAULT_CAMERA.k3;
  resetResults();
}

function hasRequiredInputs() {
  const hasImages = Boolean(state.base && state.target);
  const sameSize = hasImages && state.base.width === state.target.width && state.base.height === state.target.height;
  return Boolean(sameSize && state.refRois.length > 0 && state.wallRoi);
}

function updateReadyState() {
  const hasImages = Boolean(state.base && state.target);
  const sameSize = hasImages && state.base.width === state.target.width && state.base.height === state.target.height;
  els.analyzeBtn.disabled = !hasRequiredInputs();

  if (!hasImages) {
    setStatus("请先上传两张图片。");
  } else if (!sameSize) {
    setStatus(`两张图尺寸不一致：${state.base.width}x${state.base.height} / ${state.target.width}x${state.target.height}`);
  } else if (state.refRois.length === 0) {
    setStatus("先框固定参考区：画面中理论上不动的建筑、支架、背景等。");
  } else if (!state.wallRoi) {
    setStatus("再框墙体分析区：需要计算隆起的墙面区域。");
  } else {
    setStatus(`已设置 ${state.refRois.length} 个参考区和 1 个墙体区，可以开始计算。`);
  }
}

function setView(view) {
  state.view = view;
  els.baseTab.classList.toggle("active", view === "base");
  els.targetTab.classList.toggle("active", view === "target");
  render();
}

function populateResults(cells, summary) {
  els.summaryPanel.classList.remove("hidden");
  els.tableWrap.classList.remove("hidden");
  els.predictedBulgeValue.textContent = `${fmt(summary.predicted_bulge_mm_approx, 3)} mm`;
  els.medianValue.textContent = `${fmt(summary.corrected_wall_motion_px_median)} px`;
  els.residualP95Value.textContent = `${fmt(summary.local_residual_px_p95)} px`;
  els.stdValue.textContent = `${fmt(summary.global_ref_disp_px)} px`;
  els.globalCcValue.textContent = summary.global_cc == null ? "-" : fmt(summary.global_cc, 4);
  els.qualityValue.textContent = fmt(summary.wall_response, 3);

  els.detailRows.innerHTML = cells.map((cell) => `
    <tr>
      <td>${cell.row + 1}</td>
      <td>${cell.col + 1}</td>
      <td>${fmt(cell.dx_px, 3)}</td>
      <td>${fmt(cell.dy_px, 3)}</td>
      <td>${fmt(cell.disp_px, 4)}</td>
      <td>${fmt(cell.local_residual_disp_px, 4)}</td>
      <td>${fmt(cell.response, 3)}</td>
    </tr>
  `).join("");
}

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function resultToCsv() {
  const header = [
    "row", "col", "x0", "y0", "x1", "y1",
    "dx_px", "dy_px", "disp_px",
    "affine_pred_dx_px", "affine_pred_dy_px",
    "local_residual_dx_px", "local_residual_dy_px", "local_residual_disp_px",
    "response", "prominent"
  ];
  const lines = [header.join(",")];
  state.result.cells.forEach((cell) => {
    lines.push([
      cell.row + 1,
      cell.col + 1,
      cell.x0,
      cell.y0,
      cell.x1,
      cell.y1,
      cell.dx_px,
      cell.dy_px,
      cell.disp_px,
      cell.affine_pred_dx_px,
      cell.affine_pred_dy_px,
      cell.local_residual_dx_px,
      cell.local_residual_dy_px,
      cell.local_residual_disp_px,
      cell.response,
      cell.is_prominent ? "yes" : "no",
    ].map(csvEscape).join(","));
  });

  const s = state.result.summary;
  lines.push("");
  Object.entries(s).forEach(([key, value]) => {
    lines.push([key, Array.isArray(value) || typeof value === "object" ? JSON.stringify(value) : value].map(csvEscape).join(","));
  });
  return "\ufeff" + lines.join("\n");
}

function downloadUrl(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function analyze() {
  if (!hasRequiredInputs()) return;

  const cols = clamp(parseInt(els.colsInput.value, 10) || 4, 2, 16);
  const rows = clamp(parseInt(els.rowsInput.value, 10) || 6, 2, 16);
  let cameraParams;
  try {
    cameraParams = readCameraParams();
  } catch (err) {
    setStatus(`参数错误：${err.message}`);
    return;
  }
  const minCell = Math.min(rectWidth(state.wallRoi) / cols, rectHeight(state.wallRoi) / rows);
  if (minCell < 24) {
    setStatus("墙体分析区分块过小，请减少行列数或重新框选更大的区域。");
    return;
  }

  const form = new FormData();
  form.append("base_image", state.base.file, state.base.name);
  form.append("target_image", state.target.file, state.target.name);
  form.append("ref_rois", JSON.stringify(state.refRois));
  form.append("wall_roi", JSON.stringify(state.wallRoi));
  form.append("cols", String(cols));
  form.append("rows", String(rows));
  form.append("distance_m", String(parseFloat(els.distanceInput.value) || 25));
  form.append("camera_params", JSON.stringify(cameraParams));

  els.analyzeBtn.disabled = true;
  setStatus("正在调用 Python/OpenCV：去畸变、参考区 ECC 配准、墙体分块计算...");

  let succeeded = false;
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "后端计算失败");
    }

    const annotatedImage = await imageFromUrl(`data:image/png;base64,${payload.annotated_png_base64}`);
    state.result = {
      summary: payload.summary,
      cells: payload.cells,
      annotatedImage,
      annotatedPngBase64: payload.annotated_png_base64,
    };
    populateResults(state.result.cells, state.result.summary);
    setView("target");
    setStatus(`计算完成：全局模型 ${payload.summary.global_model}，红色为局部残差更突出的分区。`);
    succeeded = true;
  } catch (err) {
    setStatus(`计算失败：${err.message}`);
  } finally {
    els.analyzeBtn.disabled = !hasRequiredInputs();
    if (!succeeded) updateReadyState();
  }
}

function resetResults() {
  state.result = null;
  els.summaryPanel.classList.add("hidden");
  els.tableWrap.classList.add("hidden");
  if (state.view === "target") setView("base");
}

function downloadCsv() {
  if (!state.result) return;
  const blob = new Blob([resultToCsv()], { type: "text/csv;charset=utf-8" });
  downloadUrl(URL.createObjectURL(blob), "严谨隆起分析结果.csv");
}

function downloadPng() {
  if (!state.result) return;
  downloadUrl(`data:image/png;base64,${state.result.annotatedPngBase64}`, "严谨隆起分析标注图.png");
}

els.baseInput.addEventListener("change", async (evt) => {
  const file = evt.target.files?.[0];
  if (!file) return;
  state.base = await loadImageFile(file);
  state.refRois = [];
  state.wallRoi = null;
  resetResults();
  setView("base");
  updateReadyState();
});

els.targetInput.addEventListener("change", async (evt) => {
  const file = evt.target.files?.[0];
  if (!file) return;
  state.target = await loadImageFile(file);
  resetResults();
  updateReadyState();
});

els.canvas.addEventListener("pointerdown", (evt) => {
  if (state.view !== "base" || !state.base) return;
  const p = getCanvasPoint(evt);
  state.dragStart = { x: p.x, y: p.y, current: p };
  els.canvas.setPointerCapture(evt.pointerId);
});

els.canvas.addEventListener("pointermove", (evt) => {
  if (!state.dragStart) return;
  state.dragStart.current = getCanvasPoint(evt);
  render();
});

els.canvas.addEventListener("pointerup", (evt) => {
  if (!state.dragStart) return;
  const rect = normalizeRect(state.dragStart, getCanvasPoint(evt));
  state.dragStart = null;

  if (rectWidth(rect) >= 40 && rectHeight(rect) >= 40) {
    const roi = {
      x0: Math.round(rect.x0),
      y0: Math.round(rect.y0),
      x1: Math.round(rect.x1),
      y1: Math.round(rect.y1),
    };
    if (state.drawMode === "ref") {
      state.refRois.push(roi);
    } else {
      state.wallRoi = roi;
    }
    resetResults();
  }

  updateReadyState();
  render();
});

els.refModeBtn.addEventListener("click", () => setDrawMode("ref"));
els.wallModeBtn.addEventListener("click", () => setDrawMode("wall"));

els.useDefaultRoi.addEventListener("click", () => {
  if (!state.base) {
    setStatus("请先上传基准图。");
    return;
  }
  state.refRois = [{
    x0: clamp(DEFAULT_REF_ROI_25M.x0, 0, state.base.width - 1),
    y0: clamp(DEFAULT_REF_ROI_25M.y0, 0, state.base.height - 1),
    x1: clamp(DEFAULT_REF_ROI_25M.x1, 1, state.base.width),
    y1: clamp(DEFAULT_REF_ROI_25M.y1, 1, state.base.height),
  }];
  state.wallRoi = {
    x0: clamp(DEFAULT_WALL_ROI_25M.x0, 0, state.base.width - 1),
    y0: clamp(DEFAULT_WALL_ROI_25M.y0, 0, state.base.height - 1),
    x1: clamp(DEFAULT_WALL_ROI_25M.x1, 1, state.base.width),
    y1: clamp(DEFAULT_WALL_ROI_25M.y1, 1, state.base.height),
  };
  resetResults();
  setView("base");
  updateReadyState();
  render();
});

els.clearRoi.addEventListener("click", () => {
  state.refRois = [];
  state.wallRoi = null;
  resetResults();
  updateReadyState();
  render();
});

els.analyzeBtn.addEventListener("click", analyze);
els.baseTab.addEventListener("click", () => setView("base"));
els.targetTab.addEventListener("click", () => setView("target"));
els.downloadCsv.addEventListener("click", downloadCsv);
els.downloadPng.addEventListener("click", downloadPng);

["colsInput", "rowsInput", "distanceInput"].forEach((id) => {
  els[id].addEventListener("change", resetResults);
});
["fxInput", "fyInput", "cxInput", "cyInput", "k1Input", "k2Input", "p1Input", "p2Input", "k3Input"].forEach((id) => {
  els[id].addEventListener("change", resetResults);
});
els.resetCameraBtn.addEventListener("click", resetCameraParams);

setDrawMode("ref");
updateReadyState();
