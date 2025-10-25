# hy3dgen/texpaint_util.py
import os
import hashlib
from typing import Optional

import numpy as np
import trimesh
from PIL import Image

# Optional: fast sparse ops for smoothing (recommended)
try:
    import scipy.sparse as sp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================
# Low-level color utilities
# =========================
def _to_linear_rgb(rgb_255: np.ndarray) -> np.ndarray:
    x = np.clip(rgb_255.astype(np.float32) / 255.0, 0, 1)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _to_srgb(lin: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(lin, 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)

def _normalize01(x: np.ndarray, eps=1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)

def boost_colors(colors_u8: np.ndarray,
                 sat: float = 1.30,
                 contrast: float = 1.10,
                 gamma: float = 1.00) -> np.ndarray:
    # HSV sat
    rgb01 = colors_u8.astype(np.float32) / 255.0
    h, s, v = _rgb_to_hsv01(rgb01)
    s = np.clip(s * sat, 0.0, 1.0)
    rgb01 = _hsv01_to_rgb(h, s, v)

    lin = _to_linear_rgb((rgb01 * 255.0).astype(np.uint8)).astype(np.float32)
    lin = 0.5 + (lin - 0.5) * contrast
    if abs(gamma - 1.0) > 1e-3:
        lin = np.power(np.clip(lin, 0.0, 1.0), gamma)
    return _to_srgb(np.clip(lin, 0.0, 1.0))

def _rgb_to_hsv01(rgb01: np.ndarray):
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    maxc = np.max(rgb01, axis=-1)
    minc = np.min(rgb01, axis=-1)
    v = maxc
    d = maxc - minc
    s = np.where(maxc > 0, d / (maxc + 1e-8), 0.0)
    rc = (maxc - r) / (d + 1e-8)
    gc = (maxc - g) / (d + 1e-8)
    bc = (maxc - b) / (d + 1e-8)
    h = np.zeros_like(maxc)
    h = np.where(r == maxc, (bc - gc), h)
    h = np.where(g == maxc, 2.0 + (rc - bc), h)
    h = np.where(b == maxc, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0
    h = np.where(d < 1e-8, 0.0, h)
    return h, s, v

def _hsv01_to_rgb(h, s, v):
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# =========================
# Palette extraction & map
# =========================
def extract_palette_center_weighted(
    image_path: str,
    k: int = 4,
    iters: int = 18,
    seed: int = 1234,
    samples: int = 20000,
    center_sigma: float = 0.25,
    center_bias: float = 0.85,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    H, W, _ = arr.shape

    ys = (np.arange(H) + 0.5) / H * 2 - 1
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    X, Y = np.meshgrid(xs, ys)
    R2 = X * X + Y * Y

    sigma2 = max(center_sigma, 1e-3) ** 2
    w_center = np.exp(-0.5 * R2 / sigma2)
    w_center = (w_center / (w_center.sum() + 1e-8)).astype(np.float64)

    w_uniform = np.full_like(w_center, 1.0 / (H * W), dtype=np.float64)
    w_mix = center_bias * w_center + (1.0 - center_bias) * w_uniform
    w_mix = w_mix.reshape(-1)
    w_mix /= w_mix.sum()

    idx = rng.choice(H * W, size=min(samples, H * W), replace=False, p=w_mix)
    px = arr.reshape(-1, 3)[idx]

    data = _to_linear_rgb(px)
    k = max(1, min(k, len(data)))
    centers = data[rng.choice(len(data), size=k, replace=False)]
    for _ in range(iters):
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(axis=1)
        for j in range(k):
            sel = data[lab == j]
            if len(sel) > 0:
                centers[j] = sel.mean(axis=0)

    Ylin = 0.2126 * centers[:, 0] + 0.7152 * centers[:, 1] + 0.0722 * centers[:, 2]
    order = np.argsort(Ylin)
    centers = centers[order]
    return _to_srgb(centers)

def map_center_palette_colors(
    palette_rgb: np.ndarray,
    field01: np.ndarray,
    variation_strength: float = 0.15,
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    K = palette_rgb.shape[0]
    pal_lin = _to_linear_rgb(palette_rgb)
    Y = 0.2126 * pal_lin[:, 0] + 0.7152 * pal_lin[:, 1] + 0.0722 * pal_lin[:, 2]
    base_idx = np.argsort(Y)[K // 2]
    base = pal_lin[base_idx]
    others = [i for i in range(K) if i != base_idx]
    if len(others) == 0:
        return _to_srgb(np.broadcast_to(base, (len(field01), 3)))
    if len(others) == 1:
        acc1 = pal_lin[others[0]]
        acc2 = acc1
    else:
        order = np.argsort(np.abs(Y[others] - Y[base_idx]))
        acc1 = pal_lin[others[order[0]]]
        acc2 = pal_lin[others[order[1]]]
    t = np.clip(field01.astype(np.float32), 0, 1)
    acc_mix = (1.0 - t)[:, None] * acc1 + t[:, None] * acc2
    alpha = np.float32(np.clip(variation_strength, 0.0, 1.0))
    lin = (1.0 - alpha) * base[None, :] + alpha * acc_mix
    return _to_srgb(lin)


# =========================
# Fields & tri-planar
# =========================
def multi_dir_sinusoidal_fbm(P: np.ndarray, seed: int = 1234,
                             octaves: int = 4,
                             base_freq: float = 0.8,
                             lacunarity: float = 2.1,
                             gain: float = 0.55) -> np.ndarray:
    rng = np.random.RandomState(seed)
    val = np.zeros((P.shape[0],), dtype=np.float32)
    amp = 1.0
    freq = base_freq
    for _ in range(octaves):
        m = rng.randint(3, 6)
        dirs = rng.randn(m, 3).astype(np.float32)
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        phases = rng.rand(m).astype(np.float32) * 2 * np.pi
        proj = P @ (dirs.T * freq)
        s = np.sin(proj + phases)
        val += amp * (s.mean(axis=1))
        freq *= lacunarity
        amp *= gain
    return _normalize01(val)

def _prepare_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W, _ = img.shape
    uf = np.clip(u * (W - 1), 0, W - 1)
    vf = np.clip(v * (H - 1), 0, H - 1)
    u0 = np.floor(uf).astype(np.int32); v0 = np.floor(vf).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W - 1);     v1 = np.clip(v0 + 1, 0, H - 1)
    du = (uf - u0)[..., None]; dv = (vf - v0)[..., None]
    c00 = img[v0, u0].astype(np.float32); c10 = img[v0, u1].astype(np.float32)
    c01 = img[v1, u0].astype(np.float32); c11 = img[v1, u1].astype(np.float32)
    c0 = c00 * (1 - du) + c10 * du
    c1 = c01 * (1 - du) + c11 * du
    c  = c0 * (1 - dv) + c1 * dv
    return np.clip(np.round(c), 0, 255).astype(np.uint8)

def triplanar_sample_colors(
    P: np.ndarray,
    N: np.ndarray,
    img: np.ndarray,
    world_scale: float = 1.0,
    center_bias: float = 0.15,
) -> np.ndarray:
    Nn = N.astype(np.float32)
    Nn /= (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-8)
    w = np.abs(Nn); w /= (w.sum(axis=1, keepdims=True) + 1e-8)
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    s = max(1e-6, world_scale)
    Psc = P / s
    Ux = (Psc[:, 2] + 1) * 0.5; Vx = (Psc[:, 1] + 1) * 0.5
    Uy = (Psc[:, 0] + 1) * 0.5; Vy = (Psc[:, 2] + 1) * 0.5
    Uz = (Psc[:, 0] + 1) * 0.5; Vz = (Psc[:, 1] + 1) * 0.5

    if center_bias > 1e-3:
        def bias_uv(u, v, strength):
            cx, cy = 0.5, 0.5
            du = (u - cx); dv = (v - cy)
            r = np.sqrt(du * du + dv * dv)
            w = np.clip(1.0 - r / 0.707, 0, 1)
            u2 = u * (1 - strength * w) + cx * (strength * w)
            v2 = v * (1 - strength * w) + cy * (strength * w)
            return u2, v2
        Ux, Vx = bias_uv(Ux, Vx, center_bias)
        Uy, Vy = bias_uv(Uy, Vy, center_bias)
        Uz, Vz = bias_uv(Uz, Vz, center_bias)

    Cx = _bilinear_sample(img, Ux, Vx).astype(np.float32)
    Cy = _bilinear_sample(img, Uy, Vy).astype(np.float32)
    Cz = _bilinear_sample(img, Uz, Vz).astype(np.float32)
    out = (Cx * wx[:, None] + Cy * wy[:, None] + Cz * wz[:, None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

def snap_colors_toward_palette(colors_u8: np.ndarray,
                               palette_u8: np.ndarray,
                               strength: float = 0.35) -> np.ndarray:
    if strength <= 1e-6 or len(palette_u8) == 0:
        return colors_u8
    cols = _to_linear_rgb(colors_u8).astype(np.float32)
    pal  = _to_linear_rgb(palette_u8).astype(np.float32)
    d2 = ((cols[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(axis=1)
    target = pal[j]
    lin = (1 - strength) * cols + strength * target
    return _to_srgb(np.clip(lin, 0, 1))


# =========================
# Taubin smoothing
# =========================
def _normalized_adjacency(mesh: trimesh.Trimesh) -> Optional["sp.csr_matrix"]:
    if not SCIPY_OK:
        return None
    edges = mesh.edges_unique
    if edges is None or len(edges) == 0:
        F = mesh.faces
        if F is None or len(F) == 0:
            return None
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n = len(mesh.vertices)
    i = edges[:, 0].astype(np.int64); j = edges[:, 1].astype(np.int64)
    data = np.ones(len(i) * 2, dtype=np.float32)
    row = np.concatenate([i, j]); col = np.concatenate([j, i])
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel(); deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg
    Dinv = sp.diags(inv_deg.astype(np.float32))
    return Dinv @ A

def taubin_smooth_colors(mesh: trimesh.Trimesh,
                         colors: np.ndarray,
                         iters: int = 24,
                         lam: float = 0.5,
                         mu: float = -0.53) -> np.ndarray:
    C = colors.astype(np.float32, copy=False)
    N = len(mesh.vertices)
    if N >= 300_000 and iters > 10:
        iters = 10
    A_norm = _normalized_adjacency(mesh) if SCIPY_OK else None
    if A_norm is None:
        return np.clip(np.round(C), 0, 255).astype(np.uint8)

    def step(C_in: np.ndarray, alpha: float) -> np.ndarray:
        AC = A_norm @ C_in
        return C_in + alpha * (AC - C_in)

    for _ in range(iters):
        C = step(C, lam)
        C = step(C, mu)
    return np.clip(np.round(C), 0, 255).astype(np.uint8)


# =========================
# Public entry point
# =========================
def colorize_mesh_from_reference(
    mesh: trimesh.Trimesh,
    reference_image_path: str,
    seed: int = 1234,
    octaves: int = 4,
    base_freq: float = 0.8,
    smoothing_iters: int = 24,
    world_scale: float = 1.0,
    center_bias_uv: float = 0.15,
    local_weight: float = 0.60,
    palette_snap: float = 0.35
) -> trimesh.Trimesh:
    # Normals & bounds
    if getattr(mesh, "vertex_normals", None) is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.rezero(); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()

    V = mesh.vertices.astype(np.float32)
    minb, maxb = mesh.bounds
    size = (maxb - minb); size[size < 1e-6] = 1.0
    P = (V - (minb + maxb) / 2.0) / (size / 2.0)  # [-1,1]^3

    palette = extract_palette_center_weighted(reference_image_path, k=4, iters=18,
                                              seed=seed, samples=20000, center_sigma=0.25, center_bias=0.85)
    field = multi_dir_sinusoidal_fbm(P, seed=seed, octaves=octaves, base_freq=base_freq)

    base_colors = map_center_palette_colors(palette_rgb=palette, field01=field, variation_strength=0.15, seed=seed)

    N = mesh.vertex_normals.astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    AO_STRENGTH = 0.18
    ao = AO_STRENGTH * (0.5 * (1.0 - np.clip((N @ up), -1, 1)))
    base_lin = _to_linear_rgb(base_colors)
    base_lin = np.clip(base_lin * (1.0 - ao[:, None]), 0, 1)
    base_colors = _to_srgb(base_lin)

    ref_img = _prepare_image(reference_image_path)
    tri_local = triplanar_sample_colors(P, N, ref_img, world_scale=world_scale, center_bias=center_bias_uv)
    tri_local = snap_colors_toward_palette(tri_local, palette, strength=palette_snap)

    local_w = float(np.clip(local_weight, 0.0, 1.0))
    combined_lin = (1 - local_w) * _to_linear_rgb(base_colors) + local_w * _to_linear_rgb(tri_local)
    combined = _to_srgb(np.clip(combined_lin, 0, 1))

    smooth_colors = taubin_smooth_colors(mesh, combined, iters=smoothing_iters, lam=0.5, mu=-0.53)
    smooth_colors = boost_colors(smooth_colors, sat=1.35, contrast=1.12, gamma=0.98)

    mesh.visual.vertex_colors = np.concatenate(
        [smooth_colors, 255 * np.ones((len(smooth_colors), 1), dtype=np.uint8)], axis=1
    )
    return mesh
