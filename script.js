const DEFAULT_MODEL = {
  id: "default",
  name: "MobileNet Embedding + Cosine",
  size: "browser inference",
  uploadedAt: "ค่าเริ่มต้น",
  active: true,
};

const LOCAL_EMBEDDING_SIZE = 16;
const MODEL_DB_NAME = "snn-workflow-model-store";
const MODEL_DB_VERSION = 1;
const MODEL_STORE_NAME = "tfjs-model-files";

let state = {
  currentStage: 0,
  playing: false,
  timer: null,
  threshold: 0.5,
  images: [],
  pair: { leftId: null, rightId: null, type: "positive", score: 0.86 },
  models: loadModels(),
  embeddingCache: new Map(),
  modelEngine: {
    ready: false,
    loading: true,
    name: "กำลังโหลดโมเดล...",
    detail: "เตรียมระบบ embedding",
    mobilenet: null,
    customModel: null,
    customModelId: null,
    customModelKind: null,
    customInputCount: 0,
    backendModelId: null,
    backendAvailable: false,
  },
  pairRequestId: 0,
  batchResults: null,
};

function loadModels() {
  const saved = localStorage.getItem("snn-demo-models-v2");
  if (!saved) return [DEFAULT_MODEL];
  try {
    const parsed = JSON.parse(saved);
    if (!parsed.length) return [DEFAULT_MODEL];
    return dedupeModels(parsed);
  } catch {
    return [DEFAULT_MODEL];
  }
}

function saveModels() {
  state.models = dedupeModels(state.models);
  localStorage.setItem("snn-demo-models-v2", JSON.stringify(state.models));
}

function openModelDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error("IndexedDB is not available"));
      return;
    }
    const request = indexedDB.open(MODEL_DB_NAME, MODEL_DB_VERSION);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(MODEL_STORE_NAME, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveModelFiles(modelId, files) {
  const db = await openModelDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_STORE_NAME, "readwrite");
    tx.objectStore(MODEL_STORE_NAME).put({
      id: modelId,
      files: Array.from(files).map((file) => ({
        name: file.name,
        type: file.type,
        lastModified: file.lastModified,
        blob: file,
      })),
    });
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}

async function loadModelFiles(modelId) {
  const db = await openModelDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_STORE_NAME, "readonly");
    const request = tx.objectStore(MODEL_STORE_NAME).get(modelId);
    request.onsuccess = () => {
      const record = request.result;
      db.close();
      if (!record) {
        resolve(null);
        return;
      }
      resolve(record.files.map((file) => new File([file.blob], file.name, {
        type: file.type,
        lastModified: file.lastModified,
      })));
    };
    request.onerror = () => {
      db.close();
      reject(request.error);
    };
  });
}

async function clearStoredModels() {
  const db = await openModelDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(MODEL_STORE_NAME, "readwrite");
    tx.objectStore(MODEL_STORE_NAME).clear();
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}

async function loadFeatureEngine() {
  setModelEngineStatus("กำลังโหลดโมเดล...", "โหลด TensorFlow.js / MobileNet");
  await syncBackendModels();
  try {
    if (window.mobilenet) {
      const mobilenetModel = await window.mobilenet.load({ version: 2, alpha: 1.0 });
      state.modelEngine.mobilenet = mobilenetModel;
      setModelEngineStatus("MobileNet Embedding + Cosine", "MobileNet embedding + cosine similarity", true);
      if (state.models.length === 1 && state.models[0].id === "default") {
        state.models = [{ ...DEFAULT_MODEL, active: true }];
        saveModels();
      }
      return;
    }
    throw new Error("MobileNet library is not available");
  } catch (error) {
    console.warn("Using local canvas embedding fallback:", error);
    setModelEngineStatus("Canvas Embedding + Cosine", "ออฟไลน์ fallback จากพิกเซลภาพจริง", true);
    state.models = state.models.map((model) => model.id === "default" ? { ...model, name: "Canvas Embedding + Cosine" } : model);
    saveModels();
  } finally {
    renderModelList();
    const active = state.models.find((model) => model.active);
    if (active && active.id !== "default") {
      await activateModel(active.id);
      return;
    }
    if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
  }
}

async function syncBackendModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("Backend model API is unavailable");
    const data = await response.json();
    state.modelEngine.backendAvailable = true;
    const localOnly = state.models.filter((model) => !model.backend);
    const backendModels = (data.models || []).map((model) => ({ ...model, backend: true }));
    state.models = mergeModels(backendModels, localOnly);
    saveModels();
  } catch {
    state.modelEngine.backendAvailable = false;
  }
}

function mergeModels(primary, secondary) {
  const seen = new Set();
  const merged = [...primary, ...secondary].filter((model) => {
    if (seen.has(model.id)) return false;
    seen.add(model.id);
    return true;
  });
  return dedupeModels(merged);
}

function modelDedupeKey(model) {
  if (model.id === "default") return "default";
  return displayModelName(model).trim().toLowerCase();
}

function displayModelName(model) {
  return model?.originalName || model?.name || "Custom model";
}

function dedupeModels(models) {
  const byKey = new Map();
  (models || []).forEach((model) => {
    const key = modelDedupeKey(model);
    const current = byKey.get(key);
    if (!current || model.active || (!current.active && model.outputMode && !current.outputMode)) {
      byKey.set(key, model);
    }
  });
  const deduped = Array.from(byKey.values());
  if (!deduped.some((model) => model.id === "default")) {
    deduped.push({ ...DEFAULT_MODEL, active: !deduped.some((model) => model.active) });
  }
  return deduped;
}

function setModelEngineStatus(name, detail, ready = false) {
  state.modelEngine.name = name;
  state.modelEngine.detail = detail;
  state.modelEngine.ready = ready;
  state.modelEngine.loading = !ready;
  const nameEl = document.getElementById("currentModelName");
  const detailEl = document.getElementById("currentModelDetail");
  if (nameEl) nameEl.textContent = name;
  if (detailEl) detailEl.textContent = detail;
}

function makePlaceholder(label, className) {
  const colors = ["#35d6ff", "#36d477", "#ffac38", "#a78bfa", "#ff5b5b"];
  const c = colors[Math.abs(hashCode(className)) % colors.length];
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop stop-color="${c}" offset="0" />
          <stop stop-color="#0c1f38" offset="1" />
        </linearGradient>
      </defs>
      <rect width="240" height="240" rx="34" fill="url(#g)"/>
      <circle cx="120" cy="92" r="42" fill="rgba(255,255,255,.85)"/>
      <path d="M50 205c11-45 43-70 70-70s59 25 70 70" fill="rgba(255,255,255,.85)"/>
      <text x="120" y="126" text-anchor="middle" font-family="Arial" font-size="42" font-weight="800" fill="#0c1f38">${label}</text>
    </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function hashCode(str) {
  return String(str).split("").reduce((a, ch) => ((a << 5) - a + ch.charCodeAt(0)) | 0, 0);
}

function createDemoDataset() {
  const demo = [];
  ["A", "B", "C"].forEach((cls) => {
    for (let i = 1; i <= 4; i++) {
      const label = `${cls}${i}`;
      demo.push({
        id: `${cls}-${i}-${Date.now()}-${Math.random()}`,
        className: cls,
        label,
        fileName: `demo_${label}.png`,
        src: makePlaceholder(label, cls),
        demo: true,
      });
    }
  });
  state.images = demo;
  autoSelectInitialPair();
  renderAll();
  setPanel("inputPanel");
}

function groupImages() {
  return state.images.reduce((acc, img) => {
    if (!acc[img.className]) acc[img.className] = [];
    acc[img.className].push(img);
    return acc;
  }, {});
}

function getImageById(id) {
  return state.images.find((img) => img.id === id) || null;
}

function imageLabel(img) {
  if (!img) return "-";
  return `${img.label} (${img.className})`;
}

function renderImageSlot(el, img, fallback = "-") {
  if (!el) return;
  if (!img) {
    el.innerHTML = fallback;
    el.classList.remove("has-img");
    return;
  }
  el.classList.add("has-img");
  el.innerHTML = `<img src="${img.src}" alt="${img.label}" title="${imageLabel(img)}" />`;
}

function loadHtmlImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Cannot load image for embedding"));
    img.src = src;
  });
}

async function getEmbedding(imgData) {
  if (!imgData) return null;
  if (state.embeddingCache.has(imgData.id)) return state.embeddingCache.get(imgData.id);

  const htmlImage = await loadHtmlImage(imgData.src);
  let embedding;
  if (state.modelEngine.customModel && window.tf) {
    try {
      embedding = await getTfjsEmbedding(htmlImage);
    } catch (error) {
      console.warn("Custom TFJS embedding failed; using default encoder:", error);
      embedding = await getDefaultEmbedding(htmlImage);
    }
  } else {
    embedding = await getDefaultEmbedding(htmlImage);
  }
  state.embeddingCache.set(imgData.id, embedding);
  return embedding;
}

async function getDefaultEmbedding(htmlImage) {
  if (state.modelEngine.mobilenet && window.tf) {
    try {
      return await getMobileNetEmbedding(htmlImage);
    } catch (error) {
      console.warn("MobileNet embedding failed; using canvas fallback:", error);
      return getCanvasEmbedding(htmlImage);
    }
  }
  return getCanvasEmbedding(htmlImage);
}

async function getMobileNetEmbedding(htmlImage) {
  const tensor = window.tf.tidy(() => {
    const activation = state.modelEngine.mobilenet.infer(htmlImage, true);
    const normalized = window.tf.linalg.l2Normalize(activation.flatten());
    return normalized;
  });
  const values = Array.from(await tensor.data());
  tensor.dispose();
  return values;
}

async function getTfjsEmbedding(htmlImage) {
  const output = window.tf.tidy(() => {
    const input = makeImageTensor(htmlImage);
    const result = state.modelEngine.customModel.predict
      ? state.modelEngine.customModel.predict(input)
      : state.modelEngine.customModel.execute(input);
    const tensor = Array.isArray(result) ? result[0] : result;
    return window.tf.linalg.l2Normalize(tensor.flatten());
  });
  const values = Array.from(await output.data());
  output.dispose();
  return values;
}

async function getTfjsPairScore(leftImg, rightImg) {
  const [leftHtmlImage, rightHtmlImage] = await Promise.all([
    loadHtmlImage(leftImg.src),
    loadHtmlImage(rightImg.src),
  ]);
  const output = window.tf.tidy(() => {
    const leftInput = makeImageTensor(leftHtmlImage);
    const rightInput = makeImageTensor(rightHtmlImage);
    const inputs = state.modelEngine.customInputCount >= 2 ? [leftInput, rightInput] : leftInput;
    const result = state.modelEngine.customModel.predict
      ? state.modelEngine.customModel.predict(inputs)
      : state.modelEngine.customModel.execute(inputs);
    const tensor = Array.isArray(result) ? result[0] : result;
    return tensor.flatten();
  });
  const values = Array.from(await output.data());
  output.dispose();
  const outputMode = state.models.find((model) => model.id === state.modelEngine.customModelId)?.outputMode || getActiveOutputMode();
  const rawValue = values[0];
  const score = normalizeModelScore(rawValue, outputMode);
  return makeMetricResult(
    score,
    outputMode === "distance" ? "Distance" : "Similarity Score",
    outputMode === "distance" ? rawValue : score,
    outputMode,
  );
}

function makeImageTensor(htmlImage) {
  return window.tf.browser.fromPixels(htmlImage)
    .resizeBilinear([224, 224])
    .toFloat()
    .div(255)
    .expandDims(0);
}

function normalizeModelScore(value, outputMode = "auto") {
  if (!Number.isFinite(value)) return 0;
  if (outputMode === "distance") return Number((1 / (1 + Math.max(0, value))).toFixed(2));
  if (value >= 0 && value <= 1) return Number(value.toFixed(2));
  const sigmoid = 1 / (1 + Math.exp(-value));
  return Number(sigmoid.toFixed(2));
}

function getCanvasEmbedding(htmlImage) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  canvas.width = LOCAL_EMBEDDING_SIZE;
  canvas.height = LOCAL_EMBEDDING_SIZE;
  ctx.drawImage(htmlImage, 0, 0, canvas.width, canvas.height);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const values = [];
  for (let i = 0; i < pixels.length; i += 4) {
    values.push(pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255);
  }
  return normalizeVector(values);
}

function normalizeVector(values) {
  const norm = Math.hypot(...values) || 1;
  return values.map((value) => value / norm);
}

function cosineSimilarity(left, right) {
  if (!left || !right || left.length !== right.length) return 0;
  let dot = 0;
  for (let i = 0; i < left.length; i++) {
    if (!Number.isFinite(left[i]) || !Number.isFinite(right[i])) {
      throw new Error("Model embedding contains NaN or Infinity");
    }
    dot += left[i] * right[i];
  }
  const score = (dot + 1) / 2;
  if (!Number.isFinite(score)) throw new Error("Model cosine similarity is not finite");
  return Math.max(0, Math.min(1, score));
}

function renderCustomDataset() {
  const grid = document.getElementById("customClassGrid");
  const summary = document.getElementById("datasetSummary");
  const groups = groupImages();
  const classNames = Object.keys(groups).sort((a, b) => a.localeCompare(b));
  const total = state.images.length;
  summary.textContent = total ? `${classNames.length} คลาส / ${total} รูป` : "ยังไม่มีรูป — อัปโหลดรูปหรือกดโหลดตัวอย่าง Demo";
  grid.innerHTML = "";

  if (!total) {
    grid.innerHTML = `<div class="empty-state">ยังไม่มี Input Dataset<br/>เริ่มจากพิมพ์ชื่อ Class แล้วอัปโหลดรูปได้เลย</div>`;
    return;
  }

  classNames.forEach((cls) => {
    const card = document.createElement("article");
    card.className = "class-card";
    const imgs = groups[cls];
    card.innerHTML = `
      <div class="class-card-header">
        <div class="class-title">
          <div class="class-name-badge">${escapeHtml(cls)}</div>
          <div>
            <strong>Class ${escapeHtml(cls)}</strong>
            <div class="class-count">${imgs.length} รูป</div>
          </div>
        </div>
        <div>
          <button class="delete-class-btn" type="button" data-class="${escapeHtml(cls)}">ลบคลาส</button>
        </div>
      </div>
      <div class="image-grid"></div>
    `;
    card.querySelector(".delete-class-btn").addEventListener("click", () => deleteClass(cls));
    const imageGrid = card.querySelector(".image-grid");
    imgs.forEach((img) => {
      const thumb = document.createElement("div");
      thumb.className = "thumb-img";
      thumb.innerHTML = `<img src="${img.src}" alt="${escapeHtml(img.label)}"/><span>${escapeHtml(img.label)}</span>`;
      imageGrid.appendChild(thumb);
    });
    grid.appendChild(card);
  });
}

function renderMiniDataset() {
  const grid = document.getElementById("miniClassGrid");
  const groups = groupImages();
  const classNames = Object.keys(groups).sort((a, b) => a.localeCompare(b));
  grid.innerHTML = "";

  if (!state.images.length) {
    grid.innerHTML = `<div class="empty-state">ยังไม่มีรูป<br/>กลับไปที่ Input Dataset ก่อน</div>`;
    return;
  }

  classNames.forEach((cls) => {
    const row = document.createElement("div");
    row.className = "class-row";
    const label = document.createElement("div");
    label.className = "class-label";
    label.textContent = cls;
    const thumbs = document.createElement("div");
    thumbs.className = "thumbs";
    groups[cls].slice(0, 5).forEach((img) => {
      const thumb = document.createElement("div");
      thumb.className = "thumb has-img";
      thumb.innerHTML = `<img src="${img.src}" alt="${escapeHtml(img.label)}" title="${imageLabel(img)}"/>`;
      thumbs.appendChild(thumb);
    });
    row.appendChild(label);
    row.appendChild(thumbs);
    grid.appendChild(row);
  });
}

function renderPairSelects() {
  const left = document.getElementById("leftSelect");
  const right = document.getElementById("rightSelect");
  const options = state.images.map((img) => `<option value="${img.id}">${escapeHtml(img.label)} • Class ${escapeHtml(img.className)}</option>`).join("");
  left.innerHTML = options || `<option value="">ยังไม่มีรูป</option>`;
  right.innerHTML = options || `<option value="">ยังไม่มีรูป</option>`;

  if (state.pair.leftId && state.images.some((i) => i.id === state.pair.leftId)) left.value = state.pair.leftId;
  if (state.pair.rightId && state.images.some((i) => i.id === state.pair.rightId)) right.value = state.pair.rightId;
}

function setPanel(panelId) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.panel === panelId));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === panelId));
}

function setStage(index) {
  state.currentStage = index;
  document.querySelectorAll(".stage").forEach((stage, idx) => {
    stage.classList.toggle("active", idx === index);
    stage.classList.toggle("done", idx < index);
  });
  const messages = [
    "เลือกภาพ Anchor จาก Input Dataset ที่คุณอัปโหลด",
    "สร้างคู่ภาพจาก Input จริง: class เดียวกันคือ Positive / คนละ class คือ Negative",
    "ส่งภาพคู่เข้าโครงข่าย Siamese ที่แชร์น้ำหนักร่วมกัน",
    "แปลงภาพทั้งสองเป็น embedding vector",
    "คำนวณคะแนนความคล้ายคลึงแล้วเทียบกับ Threshold",
    "ตัดสินว่าเป็นคลาสเดียวกันหรือคนละคลาส",
  ];
  document.getElementById("mainExplain").textContent = messages[index] || messages[0];
}

function nextStage() {
  const next = (state.currentStage + 1) % 6;
  setStage(next);
}

function play() {
  state.playing = true;
  document.getElementById("playBtn").textContent = "⏸ หยุด Animation";
  clearInterval(state.timer);
  state.timer = setInterval(nextStage, 1500);
}

function pause() {
  state.playing = false;
  document.getElementById("playBtn").textContent = "▶ เล่น Animation";
  clearInterval(state.timer);
}

async function calculateScore(leftImg, rightImg) {
  if (!leftImg || !rightImg) return makeMetricResult(0);
  if (state.modelEngine.backendModelId) {
    return getBackendPairScore(leftImg, rightImg);
  }
  if (state.modelEngine.customModel && state.modelEngine.customInputCount >= 2) {
    return getTfjsPairScore(leftImg, rightImg);
  }
  if (leftImg.id === rightImg.id) return makeMetricResult(1);
  const [leftEmbedding, rightEmbedding] = await Promise.all([getEmbedding(leftImg), getEmbedding(rightImg)]);
  const embeddingScore = Number(cosineSimilarity(leftEmbedding, rightEmbedding).toFixed(2));
  return makeMetricResult(embeddingScore, "Embedding Similarity", embeddingScore, "embedding");
}

function makeMetricResult(score, label = "Similarity Score", displayValue = score, outputMode = "score") {
  const numericScore = Number(score);
  const numericDisplayValue = Number(displayValue);
  return {
    score: numericScore,
    label,
    displayValue: numericDisplayValue,
    outputMode,
  };
}

function assertValidMetric(metric) {
  if (!Number.isFinite(metric?.score) || !Number.isFinite(metric?.displayValue)) {
    throw new Error("Model returned a non-finite similarity value");
  }
  return metric;
}

async function getBackendPairScore(leftImg, rightImg) {
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      modelId: state.modelEngine.backendModelId,
      leftSrc: leftImg.src,
      rightSrc: rightImg.src,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Backend compare failed");
  const outputMode = data.outputMode || "score";
  const active = state.models.find((model) => model.id === state.modelEngine.backendModelId);
  if (active) {
    active.outputMode = outputMode;
    saveModels();
  }
  return makeMetricResult(
    Number(Number(data.score).toFixed(2)),
    data.displayLabel || (outputMode === "distance" ? "Distance" : "Similarity Score"),
    Number.isFinite(Number(data.displayValue)) ? Number(data.displayValue) : Number(data.score),
    outputMode,
  );
}

async function updatePair(leftId, rightId) {
  const requestId = ++state.pairRequestId;
  const leftImg = getImageById(leftId);
  const rightImg = getImageById(rightId);
  if (!leftImg || !rightImg) {
    renderImageSlot(document.getElementById("anchorMini"), null, "-");
    renderImageSlot(document.getElementById("pairLeft"), null, "-");
    renderImageSlot(document.getElementById("pairRight"), null, "-");
    renderImageSlot(document.getElementById("bigLeft"), null, "-");
    renderImageSlot(document.getElementById("bigRight"), null, "-");
    const label = getMetricLabel();
    document.getElementById("scoreText").textContent = `${label} = -`;
    renderMetricLabel(label);
    document.getElementById("bigScore").textContent = "-";
    return;
  }

  const type = leftImg.className === rightImg.className ? "positive" : "negative";
  renderImageSlot(document.getElementById("anchorMini"), leftImg);
  renderImageSlot(document.getElementById("pairLeft"), leftImg);
  renderImageSlot(document.getElementById("pairRight"), rightImg);
  renderImageSlot(document.getElementById("bigLeft"), leftImg);
  renderImageSlot(document.getElementById("bigRight"), rightImg);

  document.getElementById("pairTypeText").textContent = type === "positive" ? "Positive Pair • คลาสเดียวกัน" : "Negative Pair • คนละคลาส";
  document.getElementById("pairMeta").textContent = `${imageLabel(leftImg)} กับ ${imageLabel(rightImg)} • ${type === "positive" ? "Positive Pair" : "Negative Pair"}`;
  const pendingLabel = getMetricLabel();
  document.getElementById("scoreText").textContent = `${pendingLabel} = กำลังคำนวณ...`;
  renderMetricLabel(pendingLabel);
  document.getElementById("bigScore").textContent = "...";

  let metric = makeMetricResult(0);
  try {
    metric = assertValidMetric(await calculateScore(leftImg, rightImg));
  } catch (error) {
    console.error(error);
    const errorLabel = getMetricLabel();
    document.getElementById("scoreText").textContent = `${errorLabel} = คำนวณไม่ได้`;
    renderMetricLabel(errorLabel);
    document.getElementById("bigScore").textContent = "-";
    document.getElementById("decisionExplain").textContent = error.message || "โหลดหรืออ่านภาพไม่สำเร็จ";
    return;
  }
  if (requestId !== state.pairRequestId) return;

  const score = metric.score;
  state.pair = { leftId, rightId, type, score };
  const same = score >= state.threshold;
  const displayValue = Number.isFinite(metric.displayValue) ? metric.displayValue.toFixed(2) : "-";
  document.getElementById("scoreText").textContent = `${metric.label} = ${displayValue}`;
  document.getElementById("bigScore").textContent = displayValue;
  renderMetricLabel(metric.label);

  document.getElementById("scoreFill").style.width = `${score * 100}%`;

  const resultBadge = document.getElementById("resultBadge");
  const bigDecision = document.getElementById("bigDecision");
  [resultBadge, bigDecision].forEach((el) => {
    el.classList.toggle("same", same);
    el.classList.toggle("diff", !same);
    el.textContent = same ? "คลาสเดียวกัน" : "คนละคลาส";
  });
  document.getElementById("decisionExplain").textContent = same ? "Score ≥ T" : "Score < T";

  renderPairSelects();
}

function makeAllPairs() {
  const pairs = [];
  for (let i = 0; i < state.images.length; i++) {
    for (let j = i + 1; j < state.images.length; j++) {
      pairs.push([state.images[i], state.images[j]]);
    }
  }
  return pairs;
}

function renderBatchImageCell(img) {
  return `
    <div class="batch-img-cell">
      <img src="${img.src}" alt="${escapeHtml(img.label)}" />
      <div>
        <strong>${escapeHtml(img.label)}</strong>
        <span>Class ${escapeHtml(img.className)}</span>
      </div>
    </div>
  `;
}

function resetBatchSummary(message = "ยังไม่ได้รันทดสอบ") {
  document.getElementById("batchTestSummary").textContent = message;
  document.getElementById("sameCorrectText").textContent = "-";
  document.getElementById("diffCorrectText").textContent = "-";
  document.getElementById("accuracyText").textContent = "-";
  document.getElementById("downloadBatchCsvBtn").disabled = true;
  state.batchResults = null;
}

function renderBatchSummary(stats) {
  document.getElementById("sameCorrectText").textContent = `${stats.sameCorrect}/${stats.sameTotal}`;
  document.getElementById("diffCorrectText").textContent = `${stats.diffCorrect}/${stats.diffTotal}`;
  document.getElementById("accuracyText").textContent = `${stats.accuracy.toFixed(2)}%`;
  document.getElementById("downloadBatchCsvBtn").disabled = false;
}

async function runBatchTest() {
  const body = document.getElementById("batchTestBody");
  const summary = document.getElementById("batchTestSummary");
  const button = document.getElementById("runBatchTestBtn");
  const pairs = makeAllPairs();

  if (state.images.length < 2) {
    resetBatchSummary("ต้องมีรูปอย่างน้อย 2 รูป");
    body.innerHTML = `<tr><td colspan="3" class="empty-cell">ยังไม่มีคู่รูปให้ทดสอบ</td></tr>`;
    return;
  }

  button.disabled = true;
  button.textContent = "กำลังทดสอบ...";
  body.innerHTML = "";
  let sameCount = 0;
  let diffCount = 0;
  const stats = {
    sameTotal: 0,
    sameCorrect: 0,
    diffTotal: 0,
    diffCorrect: 0,
    accuracy: 0,
  };
  const rows = [];

  for (let index = 0; index < pairs.length; index++) {
    const [leftImg, rightImg] = pairs[index];
    summary.textContent = `กำลังทดสอบ ${index + 1}/${pairs.length} คู่`;

    let metric;
    try {
      metric = await calculateScore(leftImg, rightImg);
    } catch (error) {
      console.error(error);
      metric = makeMetricResult(0, getMetricLabel(), 0, getActiveOutputMode());
    }

    const predictedSame = metric.score >= state.threshold;
    const actualSame = leftImg.className === rightImg.className;
    const correct = predictedSame === actualSame;
    if (predictedSame) sameCount += 1;
    else diffCount += 1;
    if (actualSame) {
      stats.sameTotal += 1;
      if (correct) stats.sameCorrect += 1;
    } else {
      stats.diffTotal += 1;
      if (correct) stats.diffCorrect += 1;
    }

    rows.push({
      left: leftImg,
      right: rightImg,
      predictedSame,
      actualSame,
      correct,
      metricLabel: metric.label,
      metricValue: metric.displayValue,
    });

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${renderBatchImageCell(leftImg)}</td>
      <td>${renderBatchImageCell(rightImg)}</td>
      <td>
        <span class="batch-result ${predictedSame ? "same" : "diff"}">${predictedSame ? "คลาสเดียวกัน" : "คนละคลาส"}</span>
      </td>
    `;
    body.appendChild(row);
  }

  const totalCorrect = stats.sameCorrect + stats.diffCorrect;
  stats.accuracy = pairs.length ? (totalCorrect / pairs.length) * 100 : 0;
  state.batchResults = {
    rows,
    stats,
    totalPairs: pairs.length,
    threshold: state.threshold,
    modelName: (state.models.find((model) => model.active) || DEFAULT_MODEL).name,
    createdAt: new Date().toLocaleString("th-TH"),
  };
  summary.textContent = `ทั้งหมด ${pairs.length} คู่ • ทำนายคลาสเดียวกัน ${sameCount} คู่ • คนละคลาส ${diffCount} คู่`;
  renderBatchSummary(stats);
  button.disabled = false;
  button.textContent = "ทดสอบทุกคู่";
}

function csvCell(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function downloadBatchCsv() {
  if (!state.batchResults) {
    alert("กรุณากดทดสอบทุกคู่ก่อน");
    return;
  }
  const report = state.batchResults;
  const lines = [
    ["Report", "SNN Batch Test"].map(csvCell).join(","),
    ["Model", report.modelName].map(csvCell).join(","),
    ["Created", report.createdAt].map(csvCell).join(","),
    ["Threshold", report.threshold.toFixed(2)].map(csvCell).join(","),
    ["Total pairs", report.totalPairs].map(csvCell).join(","),
    ["Same-class correct", `${report.stats.sameCorrect}/${report.stats.sameTotal}`].map(csvCell).join(","),
    ["Different-class correct", `${report.stats.diffCorrect}/${report.stats.diffTotal}`].map(csvCell).join(","),
    ["Accuracy", `${report.stats.accuracy.toFixed(2)}%`].map(csvCell).join(","),
    "",
    ["#", "Image 1 Label", "Image 1 Class", "Image 2 Label", "Image 2 Class", "Prediction", "Ground Truth", "Correct", "Metric", "Metric Value"].map(csvCell).join(","),
    ...report.rows.map((row, index) => [
      index + 1,
      row.left.label,
      row.left.className,
      row.right.label,
      row.right.className,
      row.predictedSame ? "คลาสเดียวกัน" : "คนละคลาส",
      row.actualSame ? "คลาสเดียวกัน" : "คนละคลาส",
      row.correct ? "ถูก" : "ผิด",
      row.metricLabel,
      Number.isFinite(Number(row.metricValue)) ? Number(row.metricValue).toFixed(4) : "",
    ].map(csvCell).join(",")),
  ];
  const blob = new Blob([`\uFEFF${lines.join("\n")}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `snn-batch-test-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function updateThreshold(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return;
  state.threshold = Math.max(0, Math.min(1, parsed));
  document.getElementById("thresholdSlider").value = state.threshold;
  document.getElementById("thresholdInput").value = state.threshold.toFixed(3);
  document.getElementById("thresholdValue").textContent = state.threshold.toFixed(2);
  document.getElementById("thresholdLine").style.left = `${state.threshold * 100}%`;
  if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
}

function randomFrom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomPositivePair() {
  const groups = groupImages();
  const candidates = Object.values(groups).filter((arr) => arr.length >= 2);
  if (!candidates.length) {
    alert("ต้องมีอย่างน้อย 1 class ที่มีรูป 2 รูปขึ้นไป เพื่อสุ่ม Positive");
    return;
  }
  const imgs = randomFrom(candidates);
  const left = randomFrom(imgs);
  let right = randomFrom(imgs.filter((img) => img.id !== left.id));
  updatePair(left.id, right.id);
}

function randomNegativePair() {
  const groups = groupImages();
  const classNames = Object.keys(groups);
  if (classNames.length < 2) {
    alert("ต้องมีอย่างน้อย 2 class เพื่อสุ่ม Negative");
    return;
  }
  const leftClass = randomFrom(classNames);
  const rightClass = randomFrom(classNames.filter((c) => c !== leftClass));
  const left = randomFrom(groups[leftClass]);
  const right = randomFrom(groups[rightClass]);
  updatePair(left.id, right.id);
}

function autoSelectInitialPair() {
  if (!state.images.length) {
    state.pair = { leftId: null, rightId: null, type: "positive", score: 0 };
    return;
  }
  const groups = groupImages();
  const withTwo = Object.values(groups).find((arr) => arr.length >= 2);
  if (withTwo) {
    state.pair = { leftId: withTwo[0].id, rightId: withTwo[1].id, type: "positive", score: 0.86 };
  } else if (state.images.length >= 2) {
    state.pair = { leftId: state.images[0].id, rightId: state.images[1].id, type: "negative", score: 0.2 };
  } else {
    state.pair = { leftId: state.images[0].id, rightId: state.images[0].id, type: "positive", score: 0.96 };
  }
}

function classNameFromPath(file, fallbackClass) {
  const path = file.webkitRelativePath || "";
  if (!path) return fallbackClass;
  const parts = path.split("/").filter(Boolean);
  if (parts.length >= 3) return parts[1];
  if (parts.length === 2) return fallbackClass || parts[0];
  return fallbackClass;
}

function addImages(files, options = {}) {
  const classNameRaw = document.getElementById("classNameInput").value.trim();
  const fallbackClass = classNameRaw || "A";
  const fileList = Array.from(files || []).filter((file) => file.type.startsWith("image/"));
  if (!fileList.length) return;

  const startIndexes = {};
  let loaded = 0;
  fileList.forEach((file) => {
    const className = options.fromFolder ? classNameFromPath(file, fallbackClass) : fallbackClass;
    if (!startIndexes[className]) {
      startIndexes[className] = state.images.filter((img) => img.className === className).length + 1;
    }
    const label = `${className}${startIndexes[className]}`;
    startIndexes[className] += 1;
    const reader = new FileReader();
    reader.onload = () => {
      state.images.push({
        id: `${Date.now()}-${Math.random()}-${file.name}`,
        className,
        label,
        fileName: file.name,
        src: reader.result,
      });
      loaded += 1;
      if (loaded === fileList.length) {
        if (!state.pair.leftId || !getImageById(state.pair.leftId)) autoSelectInitialPair();
        renderAll();
        if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
      }
    };
    reader.readAsDataURL(file);
  });
}

function addEmptyClass() {
  const groups = groupImages();
  const nextLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const next = nextLetters.find((c) => !groups[c]) || `Class${Object.keys(groups).length + 1}`;
  document.getElementById("classNameInput").value = next;
  document.getElementById("classNameInput").focus();
}

function clearDataset() {
  if (!confirm("ล้างรูปทั้งหมดในหน้านี้ใช่ไหม?")) return;
  state.images = [];
  state.embeddingCache.clear();
  state.pair = { leftId: null, rightId: null, type: "positive", score: 0 };
  renderAll();
  updatePair(null, null);
}

function deleteClass(className) {
  const count = state.images.filter((img) => img.className === className).length;
  if (!count) return;
  if (!confirm(`ลบ Class ${className} และรูปทั้งหมด ${count} รูปใช่ไหม?`)) return;

  const removedIds = new Set(state.images.filter((img) => img.className === className).map((img) => img.id));
  state.images = state.images.filter((img) => img.className !== className);
  removedIds.forEach((id) => state.embeddingCache.delete(id));

  if (removedIds.has(state.pair.leftId) || removedIds.has(state.pair.rightId)) {
    autoSelectInitialPair();
  }

  renderAll();
  if (state.pair.leftId && state.pair.rightId) {
    updatePair(state.pair.leftId, state.pair.rightId);
  } else {
    updatePair(null, null);
  }
}

function updateCurrentModelCard() {
  const active = state.models.find((m) => m.active) || DEFAULT_MODEL;
  const isLoadedCustom = active.id && active.id === state.modelEngine.customModelId;
  const isBackendActive = active.backend && active.id === state.modelEngine.backendModelId;
  const name = active.id === "default" || isLoadedCustom || isBackendActive ? state.modelEngine.name : displayModelName(active);
  document.getElementById("currentModelName").textContent = name;
  document.getElementById("currentModelDetail").textContent = active.id === "default" || isLoadedCustom || isBackendActive
    ? state.modelEngine.detail
    : "อยู่ในประวัติ กดใช้โมเดลนี้เพื่อโหลดจาก browser storage";
  renderMetricLabel();
}

function getActiveOutputMode() {
  const active = state.models.find((model) => model.active);
  return active?.outputMode || "score";
}

function getMetricLabel(outputMode = getActiveOutputMode()) {
  if (outputMode === "embedding") return "Embedding Similarity";
  return outputMode === "distance" ? "Distance" : "Similarity Score";
}

function renderMetricLabel(label = getMetricLabel()) {
  const bigLabel = document.getElementById("bigScoreLabel");
  const stageTitle = document.getElementById("scoreStageTitle");
  if (bigLabel) bigLabel.textContent = label;
  if (stageTitle) stageTitle.textContent = label;
}

function renderModelList() {
  state.models = dedupeModels(state.models);
  updateCurrentModelCard();
  const list = document.getElementById("modelList");
  list.innerHTML = "";
  state.models.forEach((model) => {
    const item = document.createElement("div");
    item.className = `model-item ${model.active ? "active" : ""}`;
    item.innerHTML = `
      <div>
        <strong>${escapeHtml(displayModelName(model))}</strong>
        <small>${escapeHtml(model.size)} • ${escapeHtml(model.uploadedAt)}${model.outputMode && model.outputMode !== "auto" ? ` • output: ${escapeHtml(model.outputMode)}` : ""}</small>
      </div>
      <button data-id="${model.id}">${model.active ? "กำลังใช้อยู่" : "ใช้โมเดลนี้"}</button>
    `;
    item.querySelector("button").addEventListener("click", () => activateModel(model.id));
    list.appendChild(item);
  });
}

async function activateModel(id) {
  if (id === "default") {
    state.modelEngine.customModel = null;
    state.modelEngine.customModelId = null;
    state.modelEngine.customModelKind = null;
    state.modelEngine.customInputCount = 0;
    state.modelEngine.backendModelId = null;
    const fallbackName = state.modelEngine.mobilenet ? "MobileNet Embedding + Cosine" : "Canvas Embedding + Cosine";
    const fallbackDetail = state.modelEngine.mobilenet ? "MobileNet embedding + cosine similarity" : "ออฟไลน์ fallback จากพิกเซลภาพจริง";
    setModelEngineStatus(fallbackName, fallbackDetail, true);
  } else if (state.models.find((model) => model.id === id)?.backend) {
    const modelMeta = state.models.find((model) => model.id === id);
    setModelEngineStatus(modelMeta.name, "กำลังโหลด Keras .h5 จาก backend");
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(id)}/activate`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        await syncBackendModels();
        throw new Error(data.error || "Cannot activate backend model");
      }
      if (data.model?.outputMode) modelMeta.outputMode = data.model.outputMode;
      state.modelEngine.customModel = null;
      state.modelEngine.customModelId = null;
      state.modelEngine.customModelKind = null;
      state.modelEngine.customInputCount = 0;
      state.modelEngine.backendModelId = id;
      setModelEngineStatus(modelMeta.name, "Keras .h5 SNN ใช้ backend คำนวณ score จริง", true);
    } catch (error) {
      console.error(error);
      alert("โหลดโมเดล .h5 จาก backend ไม่สำเร็จ ให้รันเว็บด้วย python3 server.py");
      renderModelList();
      return;
    }
  } else if (id !== state.modelEngine.customModelId) {
    const modelMeta = state.models.find((model) => model.id === id);
    setModelEngineStatus(modelMeta?.name || "กำลังโหลดโมเดล...", "โหลดไฟล์โมเดลจาก browser storage");
    try {
      const files = await loadModelFiles(id);
      if (!files) {
        alert("ไม่พบไฟล์โมเดลใน browser storage แล้ว กรุณาอัปโหลด model.json พร้อม weights.bin ใหม่");
        renderModelList();
        return;
      }
      const loaded = await loadCustomTfjsModel(files);
      state.modelEngine.customModel = loaded.model;
      state.modelEngine.customModelId = id;
      state.modelEngine.customModelKind = loaded.kind;
      state.modelEngine.customInputCount = loaded.inputCount;
      state.modelEngine.backendModelId = null;
      setModelEngineStatus(modelMeta?.name || "Custom TFJS model", makeCustomModelDetail(loaded.kind, loaded.inputCount), true);
    } catch (error) {
      console.error(error);
      alert("โหลดโมเดลจากประวัติไม่สำเร็จ กรุณาอัปโหลดไฟล์ใหม่");
      renderModelList();
      return;
    }
  }
  state.embeddingCache.clear();
  state.models = state.models.map((m) => ({ ...m, active: m.id === id }));
  saveModels();
  renderModelList();
  if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
}

async function addModel(files) {
  const fileList = Array.from(files || []);
  const kerasModel = fileList.find((file) => file.name.toLowerCase().endsWith(".h5") || file.name.toLowerCase().endsWith(".keras"));
  if (kerasModel) {
    await addBackendModel(kerasModel);
    return;
  }
  const modelJson = fileList.find((file) => file.name.endsWith(".json"));
  if (!modelJson || !fileList.some((file) => file.name.endsWith(".bin"))) {
    alert("ต้องเลือกไฟล์ TensorFlow.js ให้ครบ: model.json และ weights.bin");
    return;
  }
  if (!window.tf) {
    alert("ยังโหลด TensorFlow.js ไม่สำเร็จ จึงโหลดโมเดลเองไม่ได้");
    return;
  }

  let loaded;
  try {
    loaded = await loadCustomTfjsModel(fileList);
  } catch (error) {
    console.error(error);
    alert("โหลดโมเดลไม่สำเร็จ ตรวจว่าไฟล์เป็น TensorFlow.js model.json + weights.bin และ input รับรูปขนาด 224x224x3");
    return;
  }

  const modelId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`;
  const model = {
    id: modelId,
    name: modelJson.name,
    size: `${(fileList.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(2)} MB`,
    uploadedAt: new Date().toLocaleString("th-TH"),
    active: true,
    outputMode: loaded.outputMode,
  };

  try {
    await saveModelFiles(modelId, fileList);
  } catch (error) {
    console.error(error);
    alert("โหลดโมเดลได้ แต่บันทึกไฟล์ลง browser storage ไม่สำเร็จ โมเดลนี้จะใช้ได้เฉพาะรอบนี้");
  }

  state.modelEngine.customModel = loaded.model;
  state.modelEngine.customModelId = modelId;
  state.modelEngine.customModelKind = loaded.kind;
  state.modelEngine.customInputCount = loaded.inputCount;
  setModelEngineStatus(model.name, makeCustomModelDetail(loaded.kind, loaded.inputCount), true);
  state.embeddingCache.clear();
  state.models = state.models.map((m) => ({ ...m, active: false }));
  state.models.unshift(model);
  if (!state.models.find((m) => m.id === "default")) state.models.push({ ...DEFAULT_MODEL, active: false });
  saveModels();
  renderModelList();
  if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
}

async function addBackendModel(file) {
  const formData = new FormData();
  formData.append("model", file);
  formData.append("uploadedAt", new Date().toLocaleString("th-TH"));
  let response;
  try {
    response = await fetch("/api/models", { method: "POST", body: formData });
  } catch {
    alert("อัปโหลด .h5 ไม่ได้ เพราะ backend ยังไม่ทำงาน ให้รัน python3 server.py แทน python3 -m http.server");
    return;
  }
  const data = await response.json();
  if (!response.ok) {
    alert(data.error || "อัปโหลด .h5 ไม่สำเร็จ");
    return;
  }
  const model = { ...data.model, backend: true, active: true };
  state.modelEngine.customModel = null;
  state.modelEngine.customModelId = null;
  state.modelEngine.customModelKind = null;
  state.modelEngine.customInputCount = 0;
  state.modelEngine.backendModelId = model.id;
  state.embeddingCache.clear();
  setModelEngineStatus(model.name, "Keras .h5 SNN ใช้ backend คำนวณ score จริง", true);
  state.models = state.models.map((item) => ({ ...item, active: false }));
  state.models.unshift(model);
  saveModels();
  renderModelList();
  if (state.pair.leftId && state.pair.rightId) updatePair(state.pair.leftId, state.pair.rightId);
}

async function loadCustomTfjsModel(files) {
  const fileList = Array.from(files || []);
  const modelJson = fileList.find((file) => file.name.endsWith(".json"));
  const modelFiles = [modelJson, ...fileList.filter((file) => file !== modelJson)];
  let model;
  let kind = "graph";
  try {
    model = await window.tf.loadGraphModel(window.tf.io.browserFiles(modelFiles));
  } catch (graphError) {
    try {
      model = await window.tf.loadLayersModel(window.tf.io.browserFiles(modelFiles));
      kind = "layers";
    } catch (layersError) {
      throw { graphError, layersError };
    }
  }
  const inputCount = getModelInputCount(model);
  return { model, kind, inputCount, outputMode: inferTfjsOutputMode(model, inputCount) };
}

function getModelInputCount(model) {
  if (Array.isArray(model.inputs) && model.inputs.length) return model.inputs.length;
  if (Array.isArray(model.executor?.graph?.inputs) && model.executor.graph.inputs.length) {
    return model.executor.graph.inputs.length;
  }
  return 1;
}

function inferTfjsOutputMode(model, inputCount) {
  if (inputCount < 2) return "embedding";
  const lastLayer = Array.isArray(model.layers) ? model.layers.at(-1) : null;
  const layerName = `${lastLayer?.name || ""} ${lastLayer?.getClassName?.() || ""}`.toLowerCase();
  const activationName = `${lastLayer?.activation?.getClassName?.() || lastLayer?.activation?.name || ""}`.toLowerCase();
  const outputNames = `${(model.outputs || []).map((output) => output.name || "").join(" ")}`.toLowerCase();
  const descriptor = `${layerName} ${activationName} ${outputNames}`;
  if (activationName.includes("sigmoid") || activationName.includes("softmax")) return "score";
  if (["distance", "euclidean", "manhattan", "lambda", "l1", "l2"].some((token) => descriptor.includes(token))) {
    return "distance";
  }
  return inputCount >= 2 ? "distance" : "embedding";
}

function makeCustomModelDetail(kind, inputCount) {
  if (inputCount >= 2) return `TensorFlow.js ${kind} SNN pair model: ส่งภาพสองฝั่งเข้าโมเดลโดยตรง`;
  return `TensorFlow.js ${kind} encoder model: สร้าง embedding แล้วเทียบ cosine`;
}

async function resetModels() {
  state.modelEngine.customModel = null;
  state.modelEngine.customModelId = null;
  state.modelEngine.customModelKind = null;
  state.modelEngine.customInputCount = 0;
  state.modelEngine.backendModelId = null;
  state.embeddingCache.clear();
  try {
    await fetch("/api/models", { method: "DELETE" });
  } catch (error) {
    console.warn("Cannot clear backend model files:", error);
  }
  try {
    await clearStoredModels();
  } catch (error) {
    console.warn("Cannot clear stored model files:", error);
  }
  state.models = [DEFAULT_MODEL];
  saveModels();
  renderModelList();
  activateModel("default");
}

function renderAll() {
  renderCustomDataset();
  renderMiniDataset();
  renderPairSelects();
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setPanel(tab.dataset.panel)));
  document.getElementById("playBtn").addEventListener("click", () => state.playing ? pause() : play());
  document.getElementById("resetBtn").addEventListener("click", () => { pause(); setStage(0); if (state.pair.leftId) updatePair(state.pair.leftId, state.pair.rightId); });
  document.getElementById("thresholdSlider").addEventListener("input", (e) => updateThreshold(e.target.value));
  document.getElementById("thresholdInput").addEventListener("input", (e) => updateThreshold(e.target.value));

  document.getElementById("imageUpload").addEventListener("change", (e) => {
    addImages(e.target.files);
    e.target.value = "";
  });
  document.getElementById("folderUpload").addEventListener("change", (e) => {
    addImages(e.target.files, { fromFolder: true });
    e.target.value = "";
  });
  document.getElementById("addClassBtn").addEventListener("click", addEmptyClass);
  document.getElementById("loadDemoBtn").addEventListener("click", createDemoDataset);
  document.getElementById("clearDatasetBtn").addEventListener("click", clearDataset);

  document.getElementById("applySelectedPair").addEventListener("click", () => {
    const left = document.getElementById("leftSelect").value;
    const right = document.getElementById("rightSelect").value;
    if (!left || !right) return alert("กรุณาอัปโหลดรูปก่อน");
    updatePair(left, right);
  });
  document.getElementById("randomPositive").addEventListener("click", randomPositivePair);
  document.getElementById("randomNegative").addEventListener("click", randomNegativePair);
  document.getElementById("runBatchTestBtn").addEventListener("click", runBatchTest);
  document.getElementById("downloadBatchCsvBtn").addEventListener("click", downloadBatchCsv);

  document.getElementById("modelUpload").addEventListener("change", (e) => {
    if (e.target.files?.length) addModel(e.target.files);
    e.target.value = "";
  });
  document.getElementById("useDefaultBtn").addEventListener("click", () => {
    if (!state.models.find((m) => m.id === "default")) state.models.push({ ...DEFAULT_MODEL, active: false });
    activateModel("default");
  });
  document.getElementById("clearHistoryBtn").addEventListener("click", resetModels);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

bindEvents();
createDemoDataset();
updateThreshold(0.5);
setStage(0);
renderModelList();
loadFeatureEngine();
