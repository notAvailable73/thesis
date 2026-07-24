// render.js — pure view functions: data -> DOM. No fetching, no event wiring.
import { $, h, esc, pct, clamp01 } from "./dom.js";

const ICONS = {
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>`,
  review: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>`,
  unknown: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4M12 17h.01"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>`,
  box: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8l-9-5-9 5 9 5 9-5zM3 8v8l9 5 9-5V8"/></svg>`,
};

const DECISION = {
  MATCH:   { cls: "match",   icon: ICONS.check,   title: "MATCH",
             blurb: (d) => `Identified as <strong>${esc(d.predicted_name)}</strong> — auto-accepted.` },
  REVIEW:  { cls: "review",  icon: ICONS.review,  title: "NEEDS REVIEW",
             blurb: (d) => `Best guess <strong>${esc(d.predicted_name)}</strong>, but confidence is low. Ask an operator to confirm.` },
  UNKNOWN: { cls: "unknown", icon: ICONS.unknown, title: "UNKNOWN ITEM",
             blurb: () => `Doesn't match any enrolled product. Routed to <strong>manual inspection</strong>.` },
};

// --- Product gallery -------------------------------------------------------
export function renderProducts(container, products, onDelete) {
  container.innerHTML = "";
  if (!products.length) {
    container.appendChild(h(`
      <div class="empty" style="grid-column:1/-1">
        ${ICONS.box}
        <div>No products enrolled yet.</div>
      </div>`));
    return;
  }
  for (const p of products) {
    const card = h(`
      <div class="product" title="${esc(p.name)} · ${p.n_shots} reference photo(s)">
        <div class="product__img">${p.thumbnail ? `<img src="${p.thumbnail}" alt="${esc(p.name)}"/>` : ""}</div>
        <div class="product__meta">
          <div class="product__name">${esc(p.name)}</div>
          <div class="product__shots">${p.n_shots}-shot prototype</div>
        </div>
        <button class="product__del" title="Delete">${ICONS.trash}</button>
      </div>`);
    card.querySelector(".product__del").addEventListener("click", () => onDelete(p));
    container.appendChild(card);
  }
}

// --- Detection result ------------------------------------------------------
export function renderResult(container, d, cfg) {
  const meta = DECISION[d.decision] || DECISION.UNKNOWN;
  const threshold = d.threshold ?? cfg?.uncertainty_threshold ?? 0.3;

  container.className = "fade-in";
  container.innerHTML = `
    <div class="banner banner--${meta.cls}">
      <div class="banner__icon">${meta.icon}</div>
      <div class="banner__text">
        <h3>${meta.title}</h3>
        <p>${meta.blurb(d)}</p>
      </div>
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="metric__label"><span>Confidence (evidential)</span><span>top-1 belief</span></div>
        <div class="metric__val">${pct(d.confidence)}</div>
        <div class="gauge"><div class="gauge__fill" style="width:${pct(d.confidence)};background:var(--brand)"></div></div>
      </div>
      <div class="metric">
        <div class="metric__label"><span>Uncertainty (vacuity)</span><span>OOD @ ≥${pct(threshold)}</span></div>
        <div class="metric__val">${pct(d.uncertainty)}</div>
        <div class="gauge">
          <div class="gauge__fill" style="width:${pct(d.uncertainty)};background:${d.uncertainty >= threshold ? "var(--unknown)" : "var(--review)"}"></div>
          <div class="gauge__mark" style="left:${pct(threshold)}" data-label="threshold"></div>
        </div>
      </div>
    </div>

    ${renderCompare(d)}

    <div class="meta-row">
      <span class="pill mono"><span class="dot"></span>softmax says ${pct(d.softmax_confidence)} confident</span>
      <span class="pill mono">${d.inference_ms.toFixed(0)} ms</span>
      <span class="pill mono">ResNet18 · ${esc(d.weights_status)}</span>
      <span class="pill mono" style="${d.adapter_status === 'trained' ? '' : 'opacity:.65'}">
        ${d.adapter_status === "trained" ? "B-PEFT adapter · trained" : "adapter · baseline (untrained)"}
      </span>
    </div>
  `;
}

function renderCompare(d) {
  const rows = d.scores.map((s) => `
    <div class="crow">
      <div class="crow__name">${esc(s.name)}<span class="sim">cos ${s.similarity.toFixed(2)} · e ${s.evidence.toFixed(1)}</span></div>
      <div class="bars">
        <div class="bar"><div class="bar__fill bar__fill--ev" style="width:${pct(s.probability)}"></div><span class="bar__val">${pct(s.probability)}</span></div>
        <div class="bar"><div class="bar__fill bar__fill--sm" style="width:${pct(s.softmax_probability)}"></div><span class="bar__val">${pct(s.softmax_probability)}</span></div>
      </div>
    </div>`).join("");

  return `
    <div class="compare">
      <div class="compare__legend">
        <span><span class="swatch swatch--ev"></span>Evidential probability (honest)</span>
        <span><span class="swatch swatch--sm"></span>Softmax probability (overconfident baseline)</span>
      </div>
      ${rows}
    </div>`;
}

// --- Batch simulation --------------------------------------------------------
const TALLY_META = {
  MATCH:   { label: "OK / matched",     cls: "match"   },
  REVIEW:  { label: "Needs review",     cls: "review"  },
  UNKNOWN: { label: "Unknown / defect", cls: "unknown" },
  ERROR:   { label: "Errors",           cls: "error"   },
};

function tallyRowHtml(tally) {
  return Object.entries(TALLY_META).map(([key, meta]) => `
    <div class="tally tally--${meta.cls}">
      <span class="tally__dot"></span>
      <span class="tally__val" data-tally="${key}">${tally[key] || 0}</span>
      <span class="tally__label">${meta.label}</span>
    </div>`).join("");
}

/** Full re-render of the live processing block (called once per item, before
 * its detect() call resolves — thumbnail + progress bar + tally-so-far). */
export function renderBatchLive(container, { file, index, total, tally }) {
  const pct100 = total ? Math.round((index / total) * 100) : 0;
  const url = URL.createObjectURL(file);
  container.innerHTML = `
    <div class="batch-progress">
      <div class="batch-progress__bar"><div class="batch-progress__fill" style="width:${pct100}%"></div></div>
      <div class="hint mono">Processing ${Math.min(index + 1, total)} / ${total} — ${esc(file.name)}</div>
    </div>
    <div class="batch-live-row">
      <div class="batch-live-thumb"><img alt="" /></div>
      <div class="tally-row">${tallyRowHtml(tally)}</div>
    </div>`;
  const img = container.querySelector(".batch-live-thumb img");
  img.src = url;
  img.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
}

/** Cheap patch of just the tally numbers after a detect() resolves — avoids
 * re-rendering (and re-creating the object URL for) the whole live block. */
export function updateBatchTally(container, tally) {
  for (const key of Object.keys(TALLY_META)) {
    const el = container.querySelector(`[data-tally="${key}"]`);
    if (el) el.textContent = tally[key] || 0;
  }
}

const norm = (s) => String(s ?? "").trim().toLowerCase();
const ICON_CHECK = `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>`;
const ICON_X = `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>`;

function buildConfusionMatrix(rows, enrolledNames) {
  const labeled = rows.filter((r) => r.trueLabel);
  if (!labeled.length) return null;

  const UNK = "— unknown —";
  const enrolledNorm = new Set([...enrolledNames].map(norm));
  const predRaw = (r) => (r.decision === "UNKNOWN" || r.decision === "ERROR") ? UNK : (r.predictedName || UNK);
  const colKey = (raw) => raw === UNK ? UNK : norm(raw);

  // Columns are keyed by normalized label so folder-derived ground truth
  // ("ok") and the enrolled product's own casing ("OK") collapse into one
  // column instead of two near-duplicates. Prefer the casing that actually
  // came from a prediction (the real product name) as the display label.
  const canonical = new Map(); // key -> display label
  for (const r of labeled) {
    const raw = predRaw(r);
    const key = colKey(raw);
    if (!canonical.has(key)) canonical.set(key, raw);
  }
  const trueLabels = [...new Set(labeled.map((r) => r.trueLabel))].sort();
  for (const t of trueLabels) {
    const key = norm(t);
    if (!canonical.has(key)) canonical.set(key, t); // diagonal column even if never predicted
  }
  const colLabel = (key) => canonical.get(key);

  const predKeys = [...canonical.keys()]
    .sort((a, b) => (a === UNK ? 1 : b === UNK ? -1 : colLabel(a).localeCompare(colLabel(b))));

  const matrix = {};
  for (const t of trueLabels) {
    matrix[t] = {};
    for (const k of predKeys) matrix[t][k] = { count: 0, reviewCount: 0 };
  }
  let correct = 0;
  for (const r of labeled) {
    const k = colKey(predRaw(r));
    matrix[r.trueLabel][k].count++;
    if (r.decision === "REVIEW") matrix[r.trueLabel][k].reviewCount++;
    if (isCorrect(r.trueLabel, k)) correct++;
  }

  // Case/whitespace-insensitive: enrollment names and folder-derived ground
  // truth are typed independently and commonly differ only in casing.
  function isCorrect(trueLabel, predKey) {
    return enrolledNorm.has(norm(trueLabel)) ? predKey === norm(trueLabel) : predKey === UNK;
  }

  return { trueLabels, predKeys, colLabel, matrix, UNK, accuracy: correct / labeled.length, n: labeled.length };
}

/** Heatmap: color encodes MAGNITUDE only (one hue, light->dark by count) — a
 * checkmark glyph, not a second hue, marks the correct cell so identity never
 * rides on color alone. Legend translates the ramp back to counts. */
function renderConfusionMatrix(cm, enrolledNames) {
  if (!cm) {
    return `<p class="hint" style="margin-top:12px">No ground-truth labels detected in this batch — pick a folder with per-class subfolders to see a confusion matrix.</p>`;
  }
  let maxCount = 1;
  for (const t of cm.trueLabels) for (const k of cm.predKeys) maxCount = Math.max(maxCount, cm.matrix[t][k].count);

  const header = `<th class="cmatrix__corner">true \\ predicted</th>` +
    cm.predKeys.map((k) => `<th>${esc(k === cm.UNK ? "UNKNOWN" : cm.colLabel(k))}</th>`).join("");

  const enrolledNorm = new Set([...enrolledNames].map(norm));
  const cellBg = (count) => count === 0 ? "transparent" : `rgba(59,130,246,${(0.12 + 0.72 * (count / maxCount)).toFixed(3)})`;

  const body = cm.trueLabels.map((t) => {
    const known = enrolledNorm.has(norm(t));
    const cells = cm.predKeys.map((k) => {
      const cell = cm.matrix[t][k];
      const ok = known ? k === norm(t) : k === cm.UNK;
      const reviewNote = cell.reviewCount > 0 ? `<span class="cmatrix__review">${cell.reviewCount} review</span>` : "";
      const check = (cell.count > 0 && ok) ? `<span class="cmatrix__check">${ICON_CHECK}</span>` : "";
      return `<td style="background:${cellBg(cell.count)}">${check}<span class="cmatrix__n">${cell.count || ""}</span>${reviewNote}</td>`;
    }).join("");
    return `<tr><th class="cmatrix__row">${esc(t)}${known ? "" : `<span class="cmatrix__badge" title="Not enrolled — correct answer is UNKNOWN">not enrolled</span>`}</th>${cells}</tr>`;
  }).join("");

  return `
    <div class="cmatrix-head">
      <h4>Confusion matrix</h4>
      <span class="hint">${cm.n} labelled image(s) — ${ICON_CHECK} marks the correct cell; shade = how many images landed there</span>
    </div>
    <div class="cmatrix-wrap">
      <table class="cmatrix">
        <thead><tr>${header}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <div class="cmatrix-legend">
      <span class="hint">count</span>
      <div class="cmatrix-legend__bar"></div>
      <span class="hint mono">0</span>
      <span class="hint mono">${maxCount}</span>
    </div>`;
}

/** For exactly two ground-truth classes where exactly one is an enrolled
 * ("normal") product and the other is not (the "positive"/anomaly class):
 * the classic 2x2 TP/TN/FP/FN breakdown. Returns null when that framing
 * doesn't cleanly apply (0, 1, or both/neither classes enrolled). */
function computeBinaryQuadrant(rows, enrolledNames) {
  const labeled = rows.filter((r) => r.trueLabel);
  if (!labeled.length) return null;
  const enrolledNorm = new Set([...enrolledNames].map(norm));
  const trueNorms = [...new Set(labeled.map((r) => norm(r.trueLabel)))];
  if (trueNorms.length !== 2) return null;
  const enrolledCount = trueNorms.filter((t) => enrolledNorm.has(t)).length;
  if (enrolledCount !== 1) return null;

  const normalNorm = trueNorms.find((t) => enrolledNorm.has(t));
  const positiveNorm = trueNorms.find((t) => !enrolledNorm.has(t));
  const firstWithNorm = (n) => labeled.find((r) => norm(r.trueLabel) === n).trueLabel;
  const normalLabel = firstWithNorm(normalNorm);
  const positiveLabel = firstWithNorm(positiveNorm);

  const bucket = () => ({ count: 0, uSum: 0 });
  const Q = { TP: bucket(), TN: bucket(), FP: bucket(), FN: bucket() };
  for (const r of labeled) {
    const predictedNormal = r.decision === "MATCH" && norm(r.predictedName) === normalNorm;
    const isPositiveTrue = norm(r.trueLabel) === positiveNorm;
    const key = isPositiveTrue ? (predictedNormal ? "FN" : "TP") : (predictedNormal ? "TN" : "FP");
    Q[key].count++;
    Q[key].uSum += r.uncertainty || 0;
  }
  for (const k of Object.keys(Q)) Q[k].mean = Q[k].count ? Q[k].uSum / Q[k].count : null;

  return {
    normalLabel, positiveLabel, ...Q,
    n: labeled.length,
    accuracy: (Q.TP.count + Q.TN.count) / labeled.length,
  };
}

function renderBinaryQuadrant(bq) {
  if (!bq) return "";
  const tile = (key, label, sub, statusCls, icon) => `
    <div class="quad-tile quad-tile--${statusCls}">
      <div class="quad-tile__top"><span class="quad-tile__icon">${icon}</span><span class="quad-tile__code">${key}</span></div>
      <div class="quad-tile__val">${bq[key].count}</div>
      <div class="quad-tile__label">${label}</div>
      <div class="quad-tile__sub">${sub}</div>
    </div>`;
  const uTxt = (k) => bq[k].mean == null ? "" : `mean uncertainty ${pct(bq[k].mean)}`;

  return `
    <div class="cmatrix-head">
      <h4>Binary outcome — "${esc(bq.positiveLabel)}" vs "${esc(bq.normalLabel)}"</h4>
      <span class="hint">${bq.n} labelled image(s) · positive = not confidently matched to "${esc(bq.normalLabel)}"</span>
    </div>
    <div class="quad-grid">
      ${tile("TP", `Correctly flagged "${esc(bq.positiveLabel)}"`, uTxt("TP"), "good", ICON_CHECK)}
      ${tile("TN", `Correctly passed "${esc(bq.normalLabel)}"`, uTxt("TN"), "good", ICON_CHECK)}
      ${tile("FP", `"${esc(bq.normalLabel)}" wrongly flagged`, uTxt("FP"), "warn", ICON_X)}
      ${tile("FN", `Missed "${esc(bq.positiveLabel)}" — called ${esc(bq.normalLabel)}`, uTxt("FN"), "critical", ICON_X)}
    </div>`;
}

export function renderBatchResults(container, { rows, tally, total, enrolledNames }) {
  const cm = buildConfusionMatrix(rows, enrolledNames);
  const bq = computeBinaryQuadrant(rows, enrolledNames);
  const labeled = rows.filter((r) => r.trueLabel);
  const meanUncertainty = labeled.length
    ? labeled.reduce((s, r) => s + (r.uncertainty || 0), 0) / labeled.length
    : null;

  const kpi = (label, value) => `<div class="stat-tile"><div class="stat-tile__label">${label}</div><div class="stat-tile__val">${value}</div></div>`;

  container.className = "fade-in";
  container.innerHTML = `
    <div class="tally-row tally-row--summary">${tallyRowHtml(tally)}</div>
    <div class="kpi-row">
      ${kpi("Images processed", total)}
      ${cm ? kpi("Accuracy", pct(cm.accuracy)) : ""}
      ${meanUncertainty != null ? kpi("Mean uncertainty", pct(meanUncertainty)) : ""}
    </div>
    ${renderBinaryQuadrant(bq)}
    ${renderConfusionMatrix(cm, enrolledNames)}
  `;
}

// --- Reference-photo thumbnails (register form) ----------------------------
export function renderRefThumbs(container, files, onRemove) {
  container.innerHTML = "";
  files.forEach((file, i) => {
    const url = URL.createObjectURL(file);
    const t = h(`<div class="thumb"><img src="${url}" alt=""/><button title="remove">×</button></div>`);
    t.querySelector("img").addEventListener("load", () => URL.revokeObjectURL(url));
    t.querySelector("button").addEventListener("click", () => onRemove(i));
    container.appendChild(t);
  });
}
