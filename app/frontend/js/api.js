// api.js — thin typed wrapper over the backend REST API.
// The single place that knows endpoint paths and response shapes.

async function handle(res) {
  if (res.status === 204) return null;
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const detail = (body && body.detail) || res.statusText || "Request failed";
    throw new ApiError(detail, res.status);
  }
  return body;
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const api = {
  async health() {
    return handle(await fetch("/api/health"));
  },
  async config() {
    return handle(await fetch("/api/config"));
  },
  async listProducts() {
    return handle(await fetch("/api/products"));
  },
  async registerProduct(name, files) {
    const fd = new FormData();
    fd.append("name", name);
    for (const f of files) fd.append("images", f);
    return handle(await fetch("/api/products", { method: "POST", body: fd }));
  },
  async deleteProduct(id) {
    return handle(await fetch(`/api/products/${id}`, { method: "DELETE" }));
  },
  async detect(file) {
    const fd = new FormData();
    fd.append("image", file);
    return handle(await fetch("/api/detect", { method: "POST", body: fd }));
  },
};
