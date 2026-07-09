// main.js — application controller. Owns state and wires DOM events to the API
// and the render functions. This is the only module with side effects at load.
import { api, ApiError } from "./api.js";
import { $, h, toast } from "./dom.js";
import { renderProducts, renderResult, renderRefThumbs } from "./render.js";
import { initBatch } from "./simulate.js";

const state = {
  config: null,
  products: [],
  refFiles: [],   // pending reference photos for enrollment
  queryFile: null,
};

// --- Theme ----------------------------------------------------------------
function initTheme() {
  const saved = localStorage.getItem("sentinel-theme") || "dark";
  document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("sentinel-theme", next);
  });
}

// --- Health / config ------------------------------------------------------
async function refreshHealth() {
  const pill = $("#health");
  const text = $("#health-text");
  try {
    const hw = await api.health();
    pill.classList.add("pill--ok");
    pill.classList.remove("pill--warn");
    const w = hw.weights_status === "imagenet" ? "imagenet" : "random-weights";
    const adapter = hw.adapter_status === "trained"
      ? `trained B-PEFT adapter${hw.checkpoint_val_accuracy != null ? ` (val acc ${(hw.checkpoint_val_accuracy * 100).toFixed(0)}%)` : ""}`
      : "baseline (untrained)";
    text.textContent = `online · ${hw.device} · ${w} · ${adapter}`;
    if (hw.weights_status !== "imagenet") {
      pill.classList.remove("pill--ok");
      pill.classList.add("pill--warn");
      pill.title = "Running on random weights — identification quality is degraded until ImageNet weights are cached.";
    } else if (hw.adapter_status !== "trained") {
      pill.title = "No trained B-PEFT checkpoint found — running the untrained baseline pipeline (fixed evidence constants, no meta-learned adapter).";
    } else {
      pill.title = `Trained B-PEFT adapter loaded from checkpoint (validation accuracy ${(hw.checkpoint_val_accuracy * 100).toFixed(1)}%).`;
    }
  } catch {
    pill.classList.remove("pill--ok");
    text.textContent = "offline";
  }
}

async function loadConfig() {
  try {
    state.config = await api.config();
    $("#shots-hint").textContent = `${state.config.min_shots}–${state.config.max_shots}`;
  } catch { /* non-fatal */ }
}

// --- Products -------------------------------------------------------------
async function loadProducts() {
  try {
    const data = await api.listProducts();
    state.products = data.products;
    $("#product-count").textContent = `${data.count} registered`;
    renderProducts($("#products"), state.products, onDeleteProduct);
  } catch (e) {
    toast(`Could not load products: ${e.message}`, "error");
  }
}

async function onDeleteProduct(p) {
  if (!confirm(`Delete "${p.name}"?`)) return;
  try {
    await api.deleteProduct(p.id);
    toast(`Deleted "${p.name}".`, "ok");
    await loadProducts();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, "error");
  }
}

// --- Registration form ----------------------------------------------------
function initRegister() {
  const form = $("#register-form");
  const showBtn = $("#show-register");
  const cancelBtn = $("#register-cancel");
  const dz = $("#ref-dropzone");
  const input = $("#ref-input");

  showBtn.addEventListener("click", () => {
    form.classList.remove("hidden");
    showBtn.classList.add("hidden");
    $("#product-name").focus();
  });
  cancelBtn.addEventListener("click", resetRegister);

  wireDropzone(dz, input, (files) => addRefFiles(files));
  input.addEventListener("change", () => { addRefFiles(input.files); input.value = ""; });

  form.addEventListener("submit", onRegisterSubmit);
}

function addRefFiles(fileList) {
  const max = state.config?.max_shots ?? 10;
  const imgs = [...fileList].filter((f) => f.type.startsWith("image/"));
  for (const f of imgs) {
    if (state.refFiles.length >= max) { toast(`Max ${max} photos.`, "error"); break; }
    state.refFiles.push(f);
  }
  renderRefThumbs($("#ref-thumbs"), state.refFiles, removeRefFile);
}

function removeRefFile(i) {
  state.refFiles.splice(i, 1);
  renderRefThumbs($("#ref-thumbs"), state.refFiles, removeRefFile);
}

async function onRegisterSubmit(e) {
  e.preventDefault();
  const name = $("#product-name").value.trim();
  const min = state.config?.min_shots ?? 1;
  if (!name) return toast("Enter a product name.", "error");
  if (state.refFiles.length < min) return toast(`Add at least ${min} reference photo(s).`, "error");

  const btn = $("#register-submit");
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Enrolling…`;
  try {
    const p = await api.registerProduct(name, state.refFiles);
    toast(`Enrolled "${p.name}" (${p.n_shots}-shot).`, "ok");
    resetRegister();
    await loadProducts();
  } catch (e) {
    toast(`Enrollment failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

function resetRegister() {
  state.refFiles = [];
  $("#register-form").reset();
  $("#ref-thumbs").innerHTML = "";
  $("#register-form").classList.add("hidden");
  $("#show-register").classList.remove("hidden");
}

// --- Inspection -----------------------------------------------------------
function initInspect() {
  const dz = $("#query-dropzone");
  const input = $("#query-input");
  wireDropzone(dz, input, (files) => setQuery(files[0]));
  input.addEventListener("change", () => { if (input.files[0]) setQuery(input.files[0]); input.value = ""; });
  $("#inspect-btn").addEventListener("click", runInspection);
}

function setQuery(file) {
  if (!file || !file.type.startsWith("image/")) return toast("Please choose an image.", "error");
  state.queryFile = file;
  const img = $("#query-img");
  const url = URL.createObjectURL(file);
  img.src = url;
  img.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
  $("#query-preview").style.display = "block";
  $("#inspect-btn").classList.remove("hidden");
}

async function runInspection() {
  if (!state.queryFile) return;
  if (!state.products.length) return toast("Enroll at least one product first.", "error");

  const btn = $("#inspect-btn");
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Inspecting…`;
  try {
    const d = await api.detect(state.queryFile);
    $("#result-empty").classList.add("hidden");
    const result = $("#result");
    result.classList.remove("hidden");
    renderResult(result, d, state.config);
    const wpill = $("#result-weights");
    wpill.textContent = `ResNet18 · ${d.weights_status}`;
    wpill.classList.remove("hidden");
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) toast(e.message, "error");
    else toast(`Inspection failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = original;
  }
}

// --- Shared: dropzone behaviour -------------------------------------------
function wireDropzone(dz, input, onFiles) {
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) onFiles(e.dataTransfer.files);
  });
}

// --- Bootstrap ------------------------------------------------------------
async function main() {
  initTheme();
  initRegister();
  initInspect();
  initBatch(() => state.products);
  await Promise.all([refreshHealth(), loadConfig()]);
  await loadProducts();
  setInterval(refreshHealth, 15000);
}

main();
