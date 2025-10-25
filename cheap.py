import io
import os
import sys
import zipfile
import pickle
import requests
import hashlib
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import trimesh

# Optional: fast sparse ops for smoothing (recommended)
try:
    import scipy.sparse as sp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# =========================================================
# Utils
# =========================================================
def _rgb_to_hsv01(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # rgb01: (...,3) in [0,1]
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    maxc = np.max(rgb01, axis=-1)
    minc = np.min(rgb01, axis=-1)
    v = maxc
    d = maxc - minc
    s = np.where(maxc > 0, d / (maxc + 1e-8), 0.0)

    # Hue
    rc = (maxc - r) / (d + 1e-8)
    gc = (maxc - g) / (d + 1e-8)
    bc = (maxc - b) / (d + 1e-8)
    h = np.zeros_like(maxc)
    cond_r = (r == maxc)
    cond_g = (g == maxc)
    cond_b = (b == maxc)
    h = np.where(cond_r, (bc - gc), h)
    h = np.where(cond_g, 2.0 + (rc - bc), h)
    h = np.where(cond_b, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0
    h = np.where(d < 1e-8, 0.0, h)
    return h, s, v

def _hsv01_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Returns (...,3) in [0,1]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def boost_colors(
    colors_u8: np.ndarray,
    sat: float = 1.30,       # 1.0 = no change; try 1.2–1.5
    contrast: float = 1.10,  # linear contrast around 0.5; try 1.05–1.20
    gamma: float = 1.00      # >1.0 darkens; <1.0 brightens midtones
) -> np.ndarray:
    # 1) saturation in HSV space
    rgb01 = colors_u8.astype(np.float32) / 255.0
    h, s, v = _rgb_to_hsv01(rgb01)
    s = np.clip(s * sat, 0.0, 1.0)
    rgb01 = _hsv01_to_rgb(h, s, v)

    # 2) gentle contrast in linear space for cleaner math
    lin = _to_linear_rgb((rgb01 * 255.0).astype(np.uint8)).astype(np.float32)
    lin = 0.5 + (lin - 0.5) * contrast
    if abs(gamma - 1.0) > 1e-3:
        lin = np.power(np.clip(lin, 0.0, 1.0), gamma)

    return _to_srgb(np.clip(lin, 0.0, 1.0))


def extract_palette_center_weighted(
    image_path: str,
    k: int = 4,
    iters: int = 16,
    seed: int = 1234,
    samples: int = 20000,
    center_sigma: float = 0.25,
    center_bias: float = 0.80,
) -> np.ndarray:
    """
    Like extract_palette, but samples pixels with a strong center bias.
    - center_sigma controls how quickly the weight falls off from the center (in [0..~1])
    - center_bias is the fraction of samples drawn with the center PDF (rest uniform)
    """
    rng = np.random.RandomState(seed)
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    H, W, _ = arr.shape

    # Coordinate grid in [-1,1]^2, (0,0) at image center
    ys = (np.arange(H) + 0.5) / H * 2 - 1
    xs = (np.arange(W) + 0.5) / W * 2 - 1
    X, Y = np.meshgrid(xs, ys)
    R2 = X * X + Y * Y

    # Center-weighted PDF: Gaussian on radius
    sigma2 = max(center_sigma, 1e-3) ** 2
    w_center = np.exp(-0.5 * R2 / sigma2)
    w_center = (w_center / (w_center.sum() + 1e-8)).astype(np.float64)

    # Build a joint sampling distribution that mixes center + uniform
    w_uniform = np.full_like(w_center, 1.0 / (H * W), dtype=np.float64)
    w_mix = center_bias * w_center + (1.0 - center_bias) * w_uniform
    w_mix = w_mix.reshape(-1)
    w_mix /= w_mix.sum()

    # Sample pixels by index
    idx = rng.choice(H * W, size=min(samples, H * W), replace=False, p=w_mix)
    px = arr.reshape(-1, 3)[idx]

    # K-Means in linear space (same loop as before)
    data = _to_linear_rgb(px)  # float [0..1], shape (S,3)
    k = max(1, min(k, len(data)))
    centers = data[rng.choice(len(data), size=k, replace=False)]
    for _ in range(iters):
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(axis=1)
        for j in range(k):
            sel = data[lab == j]
            if len(sel) > 0:
                centers[j] = sel.mean(axis=0)

    # Sort by luminance for stability
    Ylin = 0.2126 * centers[:, 0] + 0.7152 * centers[:, 1] + 0.0722 * centers[:, 2]
    order = np.argsort(Ylin)
    centers = centers[order]
    return _to_srgb(centers)

def map_center_palette_colors(
    palette_rgb: np.ndarray,
    field01: np.ndarray,
    variation_strength: float = 0.25,  # 0 = flat, 0.1..0.25 = gentle variation
    seed: int = 1234,
) -> np.ndarray:
    """
    Pick a dominant 'base' color from the center-biased palette and blend in
    small, smooth variation toward nearby palette colors using the provided field.
    """
    rng = np.random.RandomState(seed)
    K = palette_rgb.shape[0]
    pal_lin = _to_linear_rgb(palette_rgb)  # (K,3)
    # Choose the median-luma color as base (robust for small K)
    Y = 0.2126 * pal_lin[:, 0] + 0.7152 * pal_lin[:, 1] + 0.0722 * pal_lin[:, 2]
    base_idx = np.argsort(Y)[K // 2]
    base = pal_lin[base_idx]

    # Two accent colors (closest in luma to the base, if available)
    others = [i for i in range(K) if i != base_idx]
    if len(others) == 0:
        return _to_srgb(np.broadcast_to(base, (len(field01), 3)))
    if len(others) == 1:
        acc1 = pal_lin[others[0]]
        acc2 = pal_lin[others[0]]
    else:
        order = np.argsort(np.abs(Y[others] - Y[base_idx]))
        acc1 = pal_lin[others[order[0]]]
        acc2 = pal_lin[others[order[1]]]

    # Low-freq blend between the two accents, then mix into base
    t = np.clip(field01.astype(np.float32), 0, 1)  # (N,)
    acc_mix = (1.0 - t)[:, None] * acc1 + t[:, None] * acc2  # (N,3)

    # Final color: mostly base, with a nudge toward the accent mix
    alpha = np.float32(np.clip(variation_strength, 0.0, 1.0))
    lin = (1.0 - alpha) * base[None, :] + alpha * acc_mix
    return _to_srgb(lin)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _ensure_dir(path: str):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d: os.makedirs(d, exist_ok=True)

def _normalize01(x: np.ndarray, eps=1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)

def _to_linear_rgb(rgb_255: np.ndarray) -> np.ndarray:
    """sRGB -> linear RGB, rgb_255 uint8 or float [0..255] -> float [0..1]"""
    x = np.clip(rgb_255.astype(np.float32) / 255.0, 0, 1)
    a = 0.055
    lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return lin

def _to_srgb(lin: np.ndarray) -> np.ndarray:
    """linear RGB [0..1] -> sRGB uint8"""
    a = 0.055
    x = np.clip(lin, 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)

# =========================================================
# Palette from reference image (lightweight K-Means)
# =========================================================
def extract_palette(image_path: str, k: int = 3, iters: int = 16, seed: int = 1234) -> np.ndarray:
    """
    Returns k colors (uint8 RGB) representing a smooth palette from the image.
    Uses k-means in linear RGB for perceptually nicer interpolation without heavy libs.
    """
    img = Image.open(image_path).convert("RGB")
    # Downscale for speed, keep aspect
    MAXP = 64 * 64
    if img.width * img.height > MAXP:
        scale = (MAXP / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    px = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    data = _to_linear_rgb(px)  # float [0..1]

    rng = np.random.RandomState(seed)
    # init by random samples
    if len(data) < k:
        k = max(1, len(data))
    centers = data[rng.choice(len(data), size=k, replace=False)]
    for _ in range(iters):
        # assign
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        idx = d2.argmin(axis=1)
        # update
        for j in range(k):
            sel = data[idx == j]
            if len(sel) > 0:
                centers[j] = sel.mean(axis=0)
    # sort by luminance-ish (linear Y approximation)
    Y = 0.2126 * centers[:, 0] + 0.7152 * centers[:, 1] + 0.0722 * centers[:, 2]
    order = np.argsort(Y)
    centers = centers[order]
    return _to_srgb(centers)

def interpolate_palette(palette_rgb: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Piecewise linear interpolation across palette stops.
    palette_rgb: (K,3) uint8
    t: (...,) float in [0,1]
    returns (...,3) uint8
    """
    K = palette_rgb.shape[0]
    if K == 1:
        return np.broadcast_to(palette_rgb[0], t.shape + (3,))
    # convert to linear for smoothness
    pal_lin = _to_linear_rgb(palette_rgb)
    # position of each stop in [0,1]
    stops = np.linspace(0.0, 1.0, K, dtype=np.float32)
    t = np.clip(t.astype(np.float32), 0, 1)
    # find segment
    seg = np.clip(np.searchsorted(stops, t, side='right') - 1, 0, K - 2)
    t0 = stops[seg]
    t1 = stops[seg + 1]
    w = (t - t0) / np.maximum(t1 - t0, 1e-8)
    c0 = pal_lin[seg]
    c1 = pal_lin[seg + 1]
    lin = (1 - w[..., None]) * c0 + w[..., None] * c1
    return _to_srgb(lin)

# =========================================================
# Smooth 3D color field (no textures, no tiling)
# =========================================================
def multi_dir_sinusoidal_fbm(P: np.ndarray, seed: int = 1234,
                             octaves: int = 4,
                             base_freq: float = 1.0,
                             lacunarity: float = 2.0,
                             gain: float = 0.5) -> np.ndarray:
    """
    Fast smooth scalar field in 3D: sum of sin(dot(P, dir_i)*freq + phase)
    with random directions per octave. Returns value in [0,1].
    """
    rng = np.random.RandomState(seed)
    val = np.zeros((P.shape[0],), dtype=np.float32)
    amp = 1.0
    freq = base_freq
    for _ in range(octaves):
        # 3–5 random directions per octave
        m = rng.randint(3, 6)
        dirs = rng.randn(m, 3).astype(np.float32)
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        phases = rng.rand(m).astype(np.float32) * 2 * np.pi
        proj = P @ (dirs.T * freq)  # (N,m)
        s = np.sin(proj + phases)  # (-1..1)
        val += amp * (s.mean(axis=1))
        freq *= lacunarity
        amp *= gain
    val = _normalize01(val)  # 0..1
    return val

# =========================================================
# NEW: tri-planar image sampling (localized colors)
# =========================================================
def _prepare_image(image_path: str) -> np.ndarray:
    """Load reference image -> uint8 ndarray (H,W,3)"""
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear sampling from img (H,W,3), UV in [0,1].
    No tiling: clamp at borders. Returns uint8 (N,3).
    """
    H, W, _ = img.shape
    uf = np.clip(u * (W - 1), 0, W - 1)
    vf = np.clip(v * (H - 1), 0, H - 1)

    u0 = np.floor(uf).astype(np.int32)
    v0 = np.floor(vf).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)

    du = (uf - u0)[..., None]
    dv = (vf - v0)[..., None]

    c00 = img[v0, u0].astype(np.float32)
    c10 = img[v0, u1].astype(np.float32)
    c01 = img[v1, u0].astype(np.float32)
    c11 = img[v1, u1].astype(np.float32)

    c0 = c00 * (1 - du) + c10 * du
    c1 = c01 * (1 - du) + c11 * du
    c = c0 * (1 - dv) + c1 * dv
    return np.clip(np.round(c), 0, 255).astype(np.uint8)

def triplanar_sample_colors(
    P: np.ndarray,    # (N,3) positions in [-1,1]^3 space
    N: np.ndarray,    # (N,3) vertex normals (unit-ish)
    img: np.ndarray,  # (H,W,3) uint8
    world_scale: float = 1.0,
    center_bias: float = 0.0,     # 0..1 extra weight toward image center
) -> np.ndarray:
    """
    Seamless, UV-less sampling of reference image using tri-planar projection.
    - Blends X/Y/Z projections by |Nx|, |Ny|, |Nz|.
    - No tiling; clamps UV at edges to avoid streaks.
    - Optional center bias increases saturation near image center.
    """
    Nn = N.astype(np.float32)
    Nn_norm = np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-8
    Nn = Nn / Nn_norm

    # Projection weights
    w = np.abs(Nn)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    # Scale/shift P -> UV per-axis; P is [-1,1], map to [0,1]
    s = max(1e-6, world_scale)
    Psc = P / s
    Ux = (Psc[:, 2] + 1) * 0.5  # onto YZ -> use Z as U, Y as V
    Vx = (Psc[:, 1] + 1) * 0.5
    Uy = (Psc[:, 0] + 1) * 0.5  # onto XZ -> use X as U, Z as V
    Vy = (Psc[:, 2] + 1) * 0.5
    Uz = (Psc[:, 0] + 1) * 0.5  # onto XY -> use X as U, Y as V
    Vz = (Psc[:, 1] + 1) * 0.5

    cx, cy = 0.5, 0.5
    if center_bias > 1e-3:
        # Apply a soft radial bias toward center by blending with center UV
        def bias_uv(u, v, strength):
            du = (u - cx)
            dv = (v - cy)
            r = np.sqrt(du * du + dv * dv)  # 0..~0.707
            w = np.clip(1.0 - r / 0.707, 0, 1)  # center heavy
            u2 = u * (1 - strength * w) + cx * (strength * w)
            v2 = v * (1 - strength * w) + cy * (strength * w)
            return u2, v2
        Ux, Vx = bias_uv(Ux, Vx, center_bias)
        Uy, Vy = bias_uv(Uy, Vy, center_bias)
        Uz, Vz = bias_uv(Uz, Vz, center_bias)

    cx = _bilinear_sample(img, Ux, Vx)  # placeholder to silence linter (not used)
    # Sample each axis
    Cx = _bilinear_sample(img, Ux, Vx).astype(np.float32)
    Cy = _bilinear_sample(img, Uy, Vy).astype(np.float32)
    Cz = _bilinear_sample(img, Uz, Vz).astype(np.float32)

    # Blend by normal weights
    out = (Cx * wx[:, None] + Cy * wy[:, None] + Cz * wz[:, None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

# =========================================================
# NEW: softly snap arbitrary colors toward a palette
# =========================================================
def snap_colors_toward_palette(
    colors_u8: np.ndarray,      # (N,3)
    palette_u8: np.ndarray,     # (K,3)
    strength: float = 0.35      # 0 = keep original; 1 = fully snap to nearest palette color
) -> np.ndarray:
    """
    Pull each color toward its nearest palette color (in linear RGB space) by 'strength'.
    Keeps image-local variation but harmonizes with your palette.
    """
    if strength <= 1e-6 or len(palette_u8) == 0:
        return colors_u8
    cols = _to_linear_rgb(colors_u8).astype(np.float32)  # (N,3)
    pal  = _to_linear_rgb(palette_u8).astype(np.float32) # (K,3)
    # nearest palette in L2
    d2 = ((cols[:, None, :] - pal[None, :, :]) ** 2).sum(-1)  # (N,K)
    j = d2.argmin(axis=1)
    target = pal[j]  # (N,3)
    lin = (1 - strength) * cols + strength * target
    return _to_srgb(np.clip(lin, 0, 1))

# =========================================================
# Fast, stable Taubin smoothing (vectorized)
# =========================================================
def _normalized_adjacency(mesh: trimesh.Trimesh) -> Optional["sp.csr_matrix"]:
    """
    Build row-normalized adjacency A_norm = D^{-1} A as a CSR matrix.
    Uses unique edges; avoids NetworkX to keep it fast & lean.
    Returns None if SciPy unavailable.
    """
    if not SCIPY_OK:
        return None

    edges = mesh.edges_unique
    if edges is None or len(edges) == 0:
        F = mesh.faces
        if F is None or len(F) == 0:
            return None
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])

    n = len(mesh.vertices)
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)

    # Undirected adjacency
    data = np.ones(len(i) * 2, dtype=np.float32)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = 1.0 / deg
    Dinv = sp.diags(inv_deg.astype(np.float32))
    A_norm = Dinv @ A
    return A_norm

def taubin_smooth_colors(mesh: trimesh.Trimesh,
                         colors: np.ndarray,
                         iters: int = 20,
                         lam: float = 0.5,
                         mu: float = -0.53) -> np.ndarray:
    """
    Fast Taubin smoothing on vertex colors using a sparse normalized adjacency.
    C_{t+1} = C_t + alpha * (A_norm @ C_t - C_t)
    Falls back to a lightweight Python loop with fewer iters if SciPy isn't available.
    Adds progress prints and auto-caps iters on large meshes to avoid long stalls.
    """
    C = colors.astype(np.float32, copy=False)
    N = len(mesh.vertices)

    # Safety: cap iterations on huge meshes so we don't "hang"
    if N >= 300_000 and iters > 10:
        print(f"   ℹ️  Large mesh detected ({N:,} verts) — capping smoothing iters to 10.")
        iters = 10

    A_norm = _normalized_adjacency(mesh) if SCIPY_OK else None

    if A_norm is None:
        # Fallback — keep it lightweight and predictable
        print("   ⚠️  SciPy not available; using fallback neighbor averaging (reduced iters).")
        iters = min(iters, 6)

        edges = mesh.edges_unique
        if edges is None or len(edges) == 0:
            F = mesh.faces
            edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])

        nbrs = [[] for _ in range(N)]
        for a, b in edges:
            nbrs[a].append(b)
            nbrs[b].append(a)

        def lap_step(C_in: np.ndarray, alpha: float) -> np.ndarray:
            C_out = C_in.copy()
            for i, vs in enumerate(nbrs):
                if not vs:
                    continue
                avg = C_in[vs].mean(axis=0)
                C_out[i] = C_in[i] + alpha * (avg - C_in[i])
            return C_out

        for t in range(iters):
            if t % 2 == 0:
                print(f"   · smoothing pass {t+1}/{iters}")
            C = lap_step(C, lam)
            C = lap_step(C, mu)
        return np.clip(np.round(C), 0, 255).astype(np.uint8)

    # Vectorized Taubin smoothing with CSR
    def taubin_step(C_in: np.ndarray, alpha: float) -> np.ndarray:
        AC = A_norm @ C_in  # (N,3)
        return C_in + alpha * (AC - C_in)

    for t in range(iters):
        if t % 2 == 0:
            print(f"   · smoothing pass {t+1}/{iters}")
        C = taubin_step(C, lam)
        C = taubin_step(C, mu)

    return np.clip(np.round(C), 0, 255).astype(np.uint8)

# =========================================================
# Colorize mesh from reference palette + localized tri-planar image sampling
# =========================================================
def colorize_mesh_from_reference(
    mesh: trimesh.Trimesh,
    reference_image_path: str,
    seed: int = 1234,
    octaves: int = 4,
    base_freq: float = 0.8,
    smoothing_iters: int = 24,
    # NEW knobs:
    world_scale: float = 1.0,       # increases/decreases "zoom" of image projection
    center_bias_uv: float = 0.15,   # 0..1, bias UVs toward image center
    local_weight: float = 0.60,     # 0..1, how much of tri-planar colors to mix in
    palette_snap: float = 0.35      # 0..1, pull tri-planar colors toward palette
) -> trimesh.Trimesh:
    """
    Extracts a palette from the reference image and paints a smooth, low-frequency,
    non-tiled vertex color field over the mesh + localized tri-planar image sampling.
    """
    # Ensure normals & bounds
    if getattr(mesh, "vertex_normals", None) is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.rezero()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

    # Normalize vertex positions to a unit-ish box for stable frequencies
    V = mesh.vertices.astype(np.float32)
    minb, maxb = mesh.bounds
    size = (maxb - minb)
    size[size < 1e-6] = 1.0
    P = (V - (minb + maxb) / 2.0) / (size / 2.0)  # [-1,1]^3

    print("   • Extracting palette…")
    palette = extract_palette_center_weighted(
        reference_image_path, k=4, iters=18, seed=seed,
        samples=20000, center_sigma=0.25, center_bias=0.85
    )

    print("   • Building smooth scalar field…")
    field = multi_dir_sinusoidal_fbm(
        P, seed=seed, octaves=octaves, base_freq=base_freq, lacunarity=2.1, gain=0.55
    )

    print("   • Mapping field to palette (global base)…")
    base_colors = map_center_palette_colors(
        palette_rgb=palette, field01=field, variation_strength=0.15, seed=seed
    )

    # Soft AO-ish darken on downward normals to add shape perception (optional)
    N = mesh.vertex_normals.astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    AO_STRENGTH = 0.18
    ao = AO_STRENGTH * (0.5 * (1.0 - np.clip((N @ up), -1, 1)))
    base_lin = _to_linear_rgb(base_colors)
    base_lin = np.clip(base_lin * (1.0 - ao[:, None]), 0, 1)
    base_colors = _to_srgb(base_lin)

    print("   • Sampling localized colors via tri-planar projection…")
    ref_img = _prepare_image(reference_image_path)
    tri_local = triplanar_sample_colors(
        P, N, ref_img, world_scale=world_scale, center_bias=center_bias_uv
    )

    # Harmonize local samples with the palette for style consistency
    tri_local = snap_colors_toward_palette(tri_local, palette, strength=palette_snap)

    # Blend global base and localized samples
    local_w = float(np.clip(local_weight, 0.0, 1.0))
    combined_lin = (1 - local_w) * _to_linear_rgb(base_colors) + local_w * _to_linear_rgb(tri_local)
    combined = _to_srgb(np.clip(combined_lin, 0, 1))

    print("   • Diffusing colors over the mesh (Taubin)…")
    smooth_colors = taubin_smooth_colors(mesh, combined, iters=smoothing_iters, lam=0.5, mu=-0.53)
    smooth_colors = boost_colors(smooth_colors, sat=1.35, contrast=1.12, gamma=0.98)

    # Set alpha=255 and assign
    mesh.visual.vertex_colors = np.concatenate(
        [smooth_colors, 255 * np.ones((len(smooth_colors), 1), dtype=np.uint8)], axis=1
    )
    return mesh

# =========================================================
# Mesh fetch from your server (text -> GLB/PKL + preview)
# =========================================================
def generate_with_image_and_mesh(
    text_prompt: str,
    api_url: str = "https://beckett-unaffiliated-unallegorically.ngrok-free.dev/generate",
    output_dir: str = "output",
    return_pickle: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 256,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "texture": False,
        "return_gemini_image": True,
        "return_pickle": return_pickle
    }
    print(f"🎨 Generating from text: '{text_prompt}'")
    print(f"   Requesting: {'Pickled mesh' if return_pickle else 'GLB file'} + preview image")
    try:
        resp = requests.post(api_url, json=payload, timeout=300)
        if resp.status_code != 200:
            print(f"❌ Server status {resp.status_code}: {resp.text}")
            return None, None
        content_type = resp.headers.get("Content-Type", "")
        mesh_path, image_path = None, None
        if "zip" in content_type:
            print("📦 Received ZIP file, extracting...")
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(output_dir)
                names = zf.namelist()
                print("   Files:", ", ".join(names))
                for n in names:
                    if n.lower().endswith((".pkl", ".glb")):
                        mesh_path = os.path.join(output_dir, n); break
                for n in names:
                    if n.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(output_dir, n); break
            if mesh_path: print(f"✅ Mesh saved: {mesh_path}")
            else:         print("⚠  No mesh found in archive")
            if image_path: print(f"🖼  Image saved: {image_path}")
            else:          print("⚠  No preview image found in archive")
            return mesh_path, image_path
        mesh_ext = ".pkl" if return_pickle else ".glb"
        mesh_path = os.path.join(output_dir, f"model{mesh_ext}")
        with open(mesh_path, "wb") as f:
            f.write(resp.content)
        print(f"✅ Mesh saved: {mesh_path}")
        return mesh_path, None
    except Exception as e:
        print(f"❌ Error calling server: {e}")
        return None, None

# =========================================================
# Load mesh (GLB/PKL)
# =========================================================
def load_mesh(mesh_path: str, return_pickle: bool) -> Optional[trimesh.Trimesh]:
    try:
        if return_pickle:
            with open(mesh_path, "rb") as f:
                mesh = pickle.load(f)
            if isinstance(mesh, trimesh.Trimesh): return mesh
            return trimesh.Trimesh(**mesh) if isinstance(mesh, dict) else None
        m = trimesh.load(mesh_path, force='mesh')
        if isinstance(m, trimesh.Scene):
            if len(m.geometry) == 0: return None
            return list(m.geometry.values())[0]
        return m
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        return None

# =========================================================
# Report & export
# =========================================================
def report_and_export(mesh_path: str, mesh: trimesh.Trimesh) -> None:
    print(f"   Vertices: {len(mesh.vertices):,}")
    print(f"   Faces:    {len(mesh.faces):,}")
    try: print(f"   Volume:   {mesh.volume:.2f}")
    except: pass
    try: print(f"   Area:     {mesh.area:.2f}")
    except: pass
    print(f"   Bounds:   {mesh.bounds.tolist()}")
    print(f"   Center:   {mesh.centroid.tolist()}")
    base = os.path.splitext(mesh_path)[0]
    for ext in ["glb", "obj", "stl", "ply"]:
        outp = f"{base}_colored.{ext}"
        try:
            mesh.export(outp); print(f"   ✅ Exported: {outp}")
        except Exception as e:
            print(f"   ❌ Failed {ext}: {e}")

# =========================================================
# Combined visualization
# =========================================================
def create_combined_visualization(image_path: Optional[str], mesh: trimesh.Trimesh, out_dir: str) -> None:
    try:
        img = Image.open(image_path).convert("RGB") if (image_path and os.path.exists(image_path)) \
              else Image.new("RGB", (512, 512), (230, 230, 230))
        width = img.width + 420
        height = max(img.height, 420)
        combined = Image.new("RGB", (width, height), "white")
        combined.paste(img, (0, 0))
        draw = ImageDraw.Draw(combined)
        x, y = img.width + 18, 18
        lines = [
            "MESH INFORMATION", "",
            f"Vertices: {len(mesh.vertices):,}",
            f"Faces: {len(mesh.faces):,}",
            f"Area: {getattr(mesh, 'area', float('nan')):.2f}",
            f"Volume: {getattr(mesh, 'volume', float('nan')):.2f}", "",
            "Bounds:",
            f"  Min: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[0][1]:.2f}, {mesh.bounds[0][2]:.2f}]",
            f"  Max: [{mesh.bounds[1][0]:.2f}, {mesh.bounds[1][1]:.2f}, {mesh.bounds[1][2]:.2f}]",
        ]
        for i, line in enumerate(lines):
            draw.text((x, y + i * 26), line, fill=(0, 0, 0))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "combined_visualization.png")
        combined.save(out_path)
        print(f"   ✅ Saved: {out_path}")
    except Exception as e:
        print(f"   ⚠  Could not create visualization: {e}")

# =========================================================
# Main pipeline
# =========================================================
def run_example(prompt: str,
                out_dir: str,
                return_pickle: bool,
                override_reference_image: Optional[str] = None,
                seed: int = 1234):
    mesh_path, image_path = generate_with_image_and_mesh(
        text_prompt=prompt, output_dir=out_dir, return_pickle=return_pickle
    )
    if mesh_path is None:
        return
    mesh = load_mesh(mesh_path, return_pickle)
    if mesh is None:
        print("⚠  Failed to load mesh"); return

    ref_image = override_reference_image if override_reference_image else image_path
    if not ref_image or not os.path.exists(ref_image):
        # fallback: neutral palette from a tiny generated swatch
        sw = Image.new("RGB", (32, 32), (140, 140, 150))
        sw_path = os.path.join(out_dir, "fallback_ref.png"); sw.save(sw_path)
        ref_image = sw_path
        print("ℹ️  Using fallback reference swatch (no image returned by server).")

    print("\n🎨 Painting smooth + localized vertex colors from reference image...")
    mesh = colorize_mesh_from_reference(
        mesh, reference_image_path=ref_image,
        seed=seed, octaves=4, base_freq=0.8, smoothing_iters=24,
        # Fine-tune these if you want tighter/looser locality
        world_scale=1.0,         # raise to zoom out (see more of image), lower to zoom in
        center_bias_uv=0.15,     # 0..1 more weight to center of image
        local_weight=0.60,       # 0..1 how much to trust local tri-planar sampling
        palette_snap=0.35        # 0..1 pull local colors toward extracted palette
    )

    print("\n💾 Exporting colored mesh in multiple formats...")
    report_and_export(mesh_path, mesh)

    print("\n📊 Creating combined visualization...")
    create_combined_visualization(ref_image, mesh, out_dir)

if __name__ == "__main__":
    print("=" * 74)
    print("🎨 Diffuse, Non-Tiled Vertex Colorization with Localized Tri-Planar Sampling")
    print("=" * 74)

    run_example(
        prompt="Crocodile (subject in center, plain background. and place the object diagonally to show maximum artifact coverage)",
        out_dir="output_glb",
        return_pickle=False,
        # If you want to force a specific reference, set override_reference_image=\"path/to/image.png\"
        override_reference_image=None,
        seed=1234
    )

    print("\n" + "=" * 74)
    print("🎉 Completed!")
