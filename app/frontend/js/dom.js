// dom.js — tiny DOM utilities (no framework). Keeps rendering code readable.

/** Query one element. */
export const $ = (sel, root = document) => root.querySelector(sel);

/** Escape text for safe innerHTML interpolation. */
export function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Create an element from an HTML string (first child). */
export function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

export const clamp01 = (x) => Math.max(0, Math.min(1, x));
export const pct = (x) => `${(clamp01(x) * 100).toFixed(0)}%`;

/** Transient toast notification. */
export function toast(message, kind = "info", ttl = 3800) {
  const host = $("#toasts");
  if (!host) return;
  const el = h(`<div class="toast toast--${kind}"></div>`);
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .2s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 200);
  }, ttl);
}
