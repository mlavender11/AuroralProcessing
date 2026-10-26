"""
Synthetic benchmark: real frames moved by a known flow, written as a normal source file.

Layout of the benchmark file (pairs interleaved, so a sweep uses ``--frames synth --times synth_t
--gap 1 --stride 2``):

    synth          (2P, H, W) uint16   a0, b0, a1, b1, ...  (same units as the source frames)
    synth_t        (2P,) float64       fake uniform times
    true_flow      (P, H, W, 2) float32  known flow a -> b, px per pair (u = +x right, v = +y down)
    case           (P,) str            motion case of each pair, e.g. "rigid15", "shear", "vortex", "realistic"
    base           (P,) int            source frame each pair was built from
    masks/lanes    (P, H, W) bool      dark lanes in frame a (the structure that carries the motion)
    masks/stars    (H, W) bool         stars and their halos (static in both frames)
    masks/bright_patch (P, H, W) bool  still glow whose brightness changes between a and b

attrs: ``settings`` (JSON), ``noise_sigma``/``noise_blur`` (added noise, source units/px), ``lo``/``hi``.
"""

from __future__ import annotations
from pathlib import Path
import json
import h5py
import numpy as np
import cv2
from scipy.ndimage import median_filter

from auroral_processing.flow.compute_flow import fit_scaler
from auroral_processing.flow.io import open_flow

FRAMES, TIMES = "synth", "synth_t"


def measure_noise(ds, idx, mask=None):
    """
    Robust per-frame noise sigma (source units) and its spatial correlation, from differences of
    consecutive frames ``idx`` -> ``idx + 1`` in smooth regions. Returns (sigma, blur): white noise
    Gaussian-blurred by ``blur`` px has the same lag-1 correlation.
    """
    sig, r1 = [], []
    for i in idx:
        a = ds[i].astype(np.float32)
        d = ds[i + 1].astype(np.float32) - a
        g = cv2.GaussianBlur(a, (0, 0), 3)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        m = grad < np.percentile(grad, 50)
        if mask is not None:
            m &= ~mask
        d = d - np.median(d[m])
        sig.append(_clipped_std(d[m]) / np.sqrt(2))
        pair = m[:, :-1] & m[:, 1:]
        r1.append(np.corrcoef(d[:, :-1][pair], d[:, 1:][pair])[0, 1])
    r = float(np.clip(np.median(r1), 0, 0.99))
    blur = float(np.sqrt(-1 / (4 * np.log(r)))) if r > 0.05 else 0.0  # Gaussian-blurred noise: r1 = exp(-1/(4 s^2))
    return float(np.median(sig)), blur


def _clipped_std(x, n_sigma=4, iterations=3):
    """Std with outliers (stars, changing aurora) clipped; unlike the MAD it isn't quantised on integer data."""
    keep = np.ones(x.shape, bool)
    for _ in range(iterations):
        s = x[keep].std()
        keep = np.abs(x - x[keep].mean()) < n_sigma * s
    return float(x[keep].std())


def correlated_noise(rng, shape, sigma, blur):
    n = rng.standard_normal(shape).astype(np.float32)
    if blur > 0:
        n = cv2.GaussianBlur(n, (0, 0), blur)
        n /= n.std()
    return n * sigma


def find_stars(mean_frame, threshold, dilate=5):
    """Compact bright spots on a time-mean frame, dilated a little."""
    hp = mean_frame - median_filter(mean_frame, 9)
    return cv2.dilate((hp > threshold).astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)


def find_lanes(img, frac=0.12, min_area=300):
    """Dark lanes: the strongest black-tophat response of the smoothed frame."""
    k = max(15, min(img.shape) // 16) | 1
    g = cv2.GaussianBlur(img, (0, 0), 2)
    bt = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (bt > np.percentile(bt, 100 * (1 - frac))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area)
    big = big[big > 0]
    return np.isin(labels, big) if len(big) else m.astype(bool)  # drop specks between stars; lanes are long


def across_lane_direction(img, lanes):
    """Unit vector across the lanes: the dominant gradient direction on them (structure tensor)."""
    g = cv2.GaussianBlur(img, (0, 0), 3)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0)[lanes], cv2.Sobel(g, cv2.CV_32F, 0, 1)[lanes]
    J = np.array([[gx @ gx, gx @ gy], [gx @ gy, gy @ gy]])
    return np.linalg.eigh(J)[1][:, -1]


def lane_weight(lanes, speed):
    """1 on the lanes, falling linearly to 0 over a distance that keeps the flow gradient below ~0.67."""
    dist = cv2.distanceTransform((~lanes).astype(np.uint8), cv2.DIST_L2, 5)
    m = np.clip(1 - dist / max(15.0, 1.5 * speed), 0, 1).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 2)


def vortex(shape, centre, radius, peak):
    """Rankine vortex (solid-body core, 1/r outside), faded out beyond ~3 radii."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    dx, dy = xx - centre[0], yy - centre[1]
    r = np.hypot(dx, dy) + 1e-6
    vt = np.where(r < radius, peak * r / radius, peak * radius / r) * np.exp(-((r / (3 * radius)) ** 2))
    return np.stack([-vt * dy / r, vt * dx / r], -1).astype(np.float32)


def backward_map(u, iterations=50):
    """
    For a forward flow ``u`` (content at x moves to x + u(x)), the backward field w with
    w(y) = u(y - w(y)), so that b(y) = a(y - w(y)). Returns (w, max residual px).
    """
    h, w_ = u.shape[:2]
    yy, xx = np.mgrid[:h, :w_].astype(np.float32)
    u = np.ascontiguousarray(u, dtype=np.float32)
    w = u.copy()
    for _ in range(iterations):
        w_new = cv2.remap(u, xx - w[..., 0], yy - w[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        step = np.abs(w_new - w).max()
        w = w_new
        if step < 1e-3:
            break
    return w, float(step)


def warp(img, w):
    h, w_ = img.shape
    yy, xx = np.mgrid[:h, :w_].astype(np.float32)
    return cv2.remap(img, xx - w[..., 0], yy - w[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def gain_field(rng, still, n, sigma, amplitude):
    """Sum of ``n`` Gaussian blobs centred on still pixels, each with a random signed gain."""
    h, w = still.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    margin = int(sigma)
    ok = still.copy()
    ok[:margin], ok[-margin:], ok[:, :margin], ok[:, -margin:] = False, False, False, False
    ys, xs = np.nonzero(ok)
    field, blobs = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    if len(ys) == 0 or amplitude[1] <= 0:
        return field, blobs
    for _ in range(n):
        j = rng.integers(len(ys))
        blob = np.exp(-((xx - xs[j]) ** 2 + (yy - ys[j]) ** 2) / (2 * sigma**2))
        field += rng.choice([-1, 1]) * rng.uniform(*amplitude) * blob
        blobs += blob
    return field, blobs


def pick_base_frames(ds, n, scaler_lo, scaler_hi, step=4):
    """``n`` frames with the most dark-lane structure, spread across the stack."""
    N = ds.shape[0]
    cand = np.arange(1, N - 1, step)
    score = []
    for i in cand:
        img = (ds[i].astype(np.float32) - scaler_lo) / (scaler_hi - scaler_lo)
        g = cv2.GaussianBlur(img, (0, 0), 2)
        k = max(15, min(img.shape) // 16) | 1
        bt = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        score.append(np.percentile(bt, 95))
    chosen = []
    for i in cand[np.argsort(score)[::-1]]:
        if all(abs(i - c) >= N / (2 * n) for c in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    return sorted(chosen)


def max_gradient(u):
    return float(max(np.abs(np.gradient(u[..., c], axis=a)).max() for c in (0, 1) for a in (0, 1)))


def _realistic_field(flow_path, base, sigma=8, max_grad=0.6):
    """A real flow, smoothed by ``sigma`` px or more until it's gentle enough not to fold over."""
    with open_flow(flow_path) as f:
        k = int(np.argmin(np.abs(f["i0"][:] - base)))
        flow = np.nan_to_num(f["flow"][k])
    while True:
        u = np.stack([cv2.GaussianBlur(flow[..., c], (0, 0), sigma, borderType=cv2.BORDER_REFLECT) for c in range(2)], -1)
        if max_gradient(u) <= max_grad or sigma >= 64:
            return u
        sigma *= 1.5


def make_benchmark(
    src,
    out,
    *,
    frames="v4",
    base_frames=None,
    n_base=4,
    speeds=(5, 15, 25),
    shear=20.0,
    vortex_peak=10.0,
    realistic_flow=None,
    brightness=(0.05, 0.15),
    noise=None,
    seed=0,
    overwrite=False,
) -> Path:
    """
    Build a synthetic benchmark from real frames of ``src[frames]``.

    For each base frame (the mean of it and its two neighbours, to reduce noise) the aurora is
    separated from the stars, moved by a known flow, and both frames get the stars back at fixed
    positions plus fresh, independent noise matched to the source (level and spatial correlation),
    so the frame-to-frame noise matches real consecutive frames. Cases per
    base frame: rigid motion across the dark lanes at each of ``speeds``, a shear (across-lane
    speed ramping 0 -> ``shear`` along the lanes), a vortex on a lane, and, if ``realistic_flow``
    (a flow file of the same frames) is given, that flow smoothed by 8 px. In the still glow,
    2-3 patches change brightness by ``brightness`` (fractional gain range) without moving.

    Parameters
    ----------
    src : str or pathlib.Path
        Source HDF5 file.
    out : str or pathlib.Path
        Benchmark file to write.
    frames : str
        Frame dataset to build from.
    base_frames : list of int, optional
        Source frames to use. Defaults to the ``n_base`` frames with the most lane structure.
    speeds : iterable of float
        Rigid across-lane speeds, px per pair.
    realistic_flow : str or pathlib.Path, optional
        Flow file whose (smoothed) flow is used as an extra, realistic case.
    brightness : (float, float) or None
        Range of the fractional brightness change of the still patches. None for no change.
    noise : (float, float), optional
        (sigma in source units, blur px) of the noise in the output frames. Measured from the source
        if not given; (0, 0) gives noise-free frames.
    overwrite : bool
        Replace an existing benchmark. Flow files from sweeps of the old one would no longer match it.

    Returns
    -------
    pathlib.Path
        The benchmark file.
    """
    src, out = Path(src), Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True (sweeps of the old benchmark won't match the new one)")
    rng = np.random.default_rng(seed)

    with h5py.File(src, "r") as f:
        ds = f[frames]
        N, H, W = ds.shape
        sc = fit_scaler(ds, 0, N)
        lo, hi = sc.lo, sc.hi
        sample = np.unique(np.linspace(0, N - 1, min(N, 50)).astype(int))
        mean = ds[sample.tolist()].astype(np.float32).mean(0)
        stars = find_stars(mean, 0.03 * (hi - lo), dilate=9)  # wide, so faint stars and halos stay put too
        if noise is None:
            noise = measure_noise(ds, np.unique(np.linspace(0, N - 2, 5).astype(int)), stars)
        if base_frames is None:
            base_frames = pick_base_frames(ds, n_base, lo, hi)
        bases = {}
        for i in base_frames:
            if not 1 <= i <= N - 2:
                raise ValueError(f"base frame {i} needs a neighbour on each side (0 < i < {N - 1})")
            bases[i] = ds[i - 1 : i + 2].astype(np.float32).mean(0)
    sigma, blur = noise

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    frames_out, truth, cases, base_idx, lanes_out, bright_out = [], [], [], [], [], []
    worst = 0.0
    for i, base in bases.items():
        aurora = base.copy()
        aurora[stars] = median_filter(base, 15)[stars]
        star_layer = base - aurora
        unit = (aurora - lo) / (hi - lo)
        lanes = find_lanes(unit)
        d = across_lane_direction(unit, lanes) * rng.choice([-1, 1])
        along = np.array([-d[1], d[0]])

        fields = {}
        for s in speeds:
            fields[f"rigid{s:g}"] = lane_weight(lanes, s)[..., None] * (s * d).astype(np.float32)
        t = xx * along[0] + yy * along[1]
        tl = t[lanes]
        ramp = np.clip((t - tl.min()) / max(np.ptp(tl), 1), 0, 1)
        fields["shear"] = (lane_weight(lanes, shear) * ramp)[..., None] * (shear * d).astype(np.float32)
        R = min(40.0, min(H, W) / 6)
        ly, lx = np.nonzero(lanes)
        inner = (lx > 1.5 * R) & (lx < W - 1.5 * R) & (ly > 1.5 * R) & (ly < H - 1.5 * R)
        ly, lx = (ly[inner], lx[inner]) if inner.any() else (ly, lx)
        c = int(np.argmin((lx - W / 2) ** 2 + (ly - H / 2) ** 2))
        fields["vortex"] = vortex((H, W), (lx[c], ly[c]), R, min(vortex_peak, 0.6 * R))
        if realistic_flow is not None:
            fields["realistic"] = _realistic_field(realistic_flow, i)

        for case, u in fields.items():
            w, residual = backward_map(u)
            worst = max(worst, residual)
            b_aurora = warp(aurora, w)
            still = np.hypot(u[..., 0], u[..., 1]) < 0.2
            gain, blobs = (
                gain_field(rng, still & ~stars, int(rng.integers(2, 4)), min(H, W) / 12, brightness)
                if brightness
                else (np.zeros((H, W), np.float32), np.zeros((H, W), np.float32))
            )
            b_aurora = lo + (b_aurora - lo) * (1 + gain)
            a = aurora + star_layer + correlated_noise(rng, (H, W), sigma, blur)
            b = b_aurora + star_layer + correlated_noise(rng, (H, W), sigma, blur)
            frames_out += [a, b]
            truth.append(u)
            cases.append(case)
            base_idx.append(i)
            lanes_out.append(lanes)
            bright_out.append((blobs > 0.5) & still)

    settings = dict(
        source=str(src.resolve()), frames=frames, base_frames=[int(i) for i in bases], speeds=list(speeds),
        shear=shear, vortex_peak=vortex_peak, realistic_flow=str(realistic_flow) if realistic_flow else None,
        brightness=list(brightness) if brightness else None, seed=seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset(FRAMES, data=np.clip(np.rint(frames_out), 0, 65535).astype(np.uint16), chunks=(1, H, W))
        f[TIMES] = np.arange(len(frames_out), dtype=np.float64)
        f.create_dataset("true_flow", data=np.array(truth, np.float32), chunks=(1, H, W, 2), compression="lzf")
        f["case"] = np.array(cases, dtype=h5py.string_dtype())
        f["base"] = np.array(base_idx)
        f.create_dataset("masks/lanes", data=np.array(lanes_out), compression="lzf")
        f["masks/stars"] = stars
        f.create_dataset("masks/bright_patch", data=np.array(bright_out), compression="lzf")
        f.attrs.update(settings=json.dumps(settings), noise_sigma=sigma, noise_blur=blur, lo=lo, hi=hi,
                       inversion_residual=worst)
    return out


def is_benchmark(path) -> bool:
    with h5py.File(path, "r") as f:
        return "true_flow" in f
