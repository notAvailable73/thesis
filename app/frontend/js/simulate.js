// simulate.js — batch "production line" simulation. Processes many images
// sequentially through the existing /api/detect endpoint (no new backend
// route needed), rendering live progress, then a confusion matrix against
// folder-inferred ground truth (if the user picked a labelled folder).
import { api } from "./api.js";
import { $, toast } from "./dom.js";
import { renderBatchLive, updateBatchTally, renderBatchResults } from "./render.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const batchState = {
  files: [],       // [{file, trueLabel: string|null}]
  running: false,
};

export function initBatch(getProducts) {
  const dz = $("#batch-dropzone");
  const folderInput = $("#batch-input-folder");
  const flatInput = $("#batch-input-flat");

  $("#batch-pick-folder").addEventListener("click", () => folderInput.click());
  $("#batch-pick-flat").addEventListener("click", () => flatInput.click());

  folderInput.addEventListener("change", () => {
    const entries = [...folderInput.files].map((f) => ({
      file: f,
      trueLabel: inferLabelFromPath(f.webkitRelativePath),
    }));
    setFiles(entries);
    folderInput.value = "";
  });
  flatInput.addEventListener("change", () => {
    setFiles([...flatInput.files]
      .filter((f) => f.type.startsWith("image/"))
      .map((f) => ({ file: f, trueLabel: null })));
    flatInput.value = "";
  });

  dz.addEventListener("click", () => flatInput.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flatInput.click(); }
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); }));
  dz.addEventListener("drop", (e) => {
    const fl = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (fl.length) setFiles(fl.map((f) => ({ file: f, trueLabel: null })));
  });

  $("#batch-run-btn").addEventListener("click", () => runBatch(getProducts()));
}

function inferLabelFromPath(relPath) {
  // "neu_cls_small/crazing/000.jpg" -> "crazing" (immediate parent folder).
  // Files dropped directly in the picked folder (no subfolder) get null.
  const parts = (relPath || "").split("/");
  return parts.length >= 3 ? parts[parts.length - 2] : null;
}

function setFiles(entries) {
  batchState.files = entries;
  const n = entries.length;
  const withLabels = entries.filter((e) => e.trueLabel).length;
  $("#batch-file-summary").textContent = n
    ? `${n} image(s) selected` + (withLabels
        ? ` · ${withLabels} with a ground-truth label (confusion matrix will be shown)`
        : " · no ground truth detected (tally only, no confusion matrix)")
    : "";
  $("#batch-run-row").classList.toggle("hidden", n === 0);
  $("#batch-results").classList.add("hidden");
}

async function runBatch(products) {
  if (batchState.running || !batchState.files.length) return;
  if (!products.length) return toast("Enroll at least one product first.", "error");

  batchState.running = true;
  const runBtn = $("#batch-run-btn");
  const original = runBtn.innerHTML;
  runBtn.disabled = true;
  runBtn.innerHTML = `<span class="spinner"></span> Running…`;

  const liveEl = $("#batch-live");
  const resultsEl = $("#batch-results");
  liveEl.classList.remove("hidden");
  resultsEl.classList.add("hidden");

  const enrolledNames = new Set(products.map((p) => p.name));
  const rows = [];
  const tally = { MATCH: 0, REVIEW: 0, UNKNOWN: 0, ERROR: 0 };
  const total = batchState.files.length;
  // Pace the "live" feel without letting large batches drag on forever.
  const pace = total > 60 ? 60 : total > 30 ? 90 : 140;

  for (let i = 0; i < total; i++) {
    const { file, trueLabel } = batchState.files[i];
    renderBatchLive(liveEl, { file, index: i, total, tally });
    try {
      const d = await api.detect(file);
      tally[d.decision] = (tally[d.decision] || 0) + 1;
      rows.push({
        trueLabel,
        predictedName: d.predicted_name,
        decision: d.decision,
        confidence: d.confidence,
        uncertainty: d.uncertainty,
      });
    } catch (e) {
      tally.ERROR++;
      rows.push({ trueLabel, predictedName: null, decision: "ERROR", confidence: 0, uncertainty: 0 });
    }
    updateBatchTally(liveEl, tally);
    await sleep(pace);
  }

  liveEl.classList.add("hidden");
  resultsEl.classList.remove("hidden");
  renderBatchResults(resultsEl, { rows, tally, total, enrolledNames });

  runBtn.disabled = false;
  runBtn.innerHTML = original;
  batchState.running = false;
  toast(`Batch simulation complete: ${total} image(s) processed.`, "ok");
}
