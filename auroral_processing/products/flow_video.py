from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import matplotlib
import imageio
import h5py
import cv2

from auroral_processing.flow.compute_flow import FrameScaler
from auroral_processing.flow.io import open_flow, read_config

KINDS = ("quiver", "speed")
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_lut(name="turbo") -> np.ndarray:
    """(256, 3) BGR uint8 lookup table from a matplotlib colormap."""
    rgb = (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()


def block_mean(flow, step):
    """
    Average a dense (H, W, 2) flow over step x step blocks.

    Returns X, Y (block centres) and u, v, each (H // step, W // step). Edge pixels that don't
    fill a whole block are dropped.
    """
    h, w = flow.shape[0] // step, flow.shape[1] // step
    blocks = flow[: h * step, : w * step].reshape(h, step, w, step, 2).mean(axis=(1, 3))
    X, Y = np.meshgrid(np.arange(w) * step + step // 2, np.arange(h) * step + step // 2)
    return X, Y, blocks[..., 0], blocks[..., 1]


def _time_label(t0, i0, i1):
    try:
        ts = datetime.fromtimestamp(float(t0), timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"
    except (OverflowError, OSError, ValueError):
        ts = f"t={t0:g}"
    return f"{ts}   frames {i0}->{i1}"


def _draw_colourbar(img, lut, vmax, ref_len=None):
    """Colour scale (0..vmax px/pair) in the bottom-left corner, plus a reference arrow of length vmax."""
    h, w = img.shape[:2]
    bar_w, bar_h, pad = min(160, w // 3), 8, 10
    box_w = bar_w + 2 * pad + (int(ref_len) + 2 * pad if ref_len else 0)
    cv2.rectangle(img, (0, h - 36), (box_w, h), (0, 0, 0), -1)
    img[h - pad - bar_h : h - pad, pad : pad + bar_w] = lut[np.linspace(0, 255, bar_w).astype(int)][None]
    cv2.putText(img, "0", (pad, h - pad - bar_h - 4), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"{vmax:.3g} px/pair", (pad + bar_w - 70, h - pad - bar_h - 4), FONT, 0.35, WHITE, 1, cv2.LINE_AA)
    if ref_len:
        x0, y0 = pad + bar_w + 2 * pad, h - pad - bar_h // 2
        cv2.arrowedLine(img, (x0, y0), (x0 + int(ref_len), y0), WHITE, 1, cv2.LINE_AA, tipLength=0.3)


def _draw_label(img, label):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 16), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


def draw_quiver(background, flow, step, arrow_scale, vmax, lut, label="", dim=0.55):
    """
    Dimmed background with one arrow per step x step block, coloured by block-mean speed.

    ``background`` is uint8 (H, W); ``arrow_scale`` is arrow length in px per px/pair of flow.
    """
    img = cv2.cvtColor((background * dim).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    X, Y, u, v = block_mean(flow, step)
    colours = lut[(np.clip(np.hypot(u, v) / vmax, 0, 1) * 255).astype(int)]
    dx, dy = u * arrow_scale, v * arrow_scale
    for x0, y0, ddx, ddy, c in zip(X.ravel(), Y.ravel(), dx.ravel(), dy.ravel(), colours.reshape(-1, 3)):
        if not np.isfinite(ddx) or (abs(ddx) < 0.5 and abs(ddy) < 0.5):
            continue
        tip = (int(round(x0 + ddx)), int(round(y0 + ddy)))
        cv2.arrowedLine(img, (int(x0), int(y0)), tip, tuple(int(z) for z in c), 1, cv2.LINE_AA, tipLength=0.3)
    _draw_colourbar(img, lut, vmax, ref_len=vmax * arrow_scale)
    if label:
        _draw_label(img, label)
    return img


def draw_speed(flow, vmax, lut, label=""):
    """|flow| heatmap, 0..vmax px/pair."""
    speed = np.nan_to_num(np.hypot(flow[..., 0], flow[..., 1]))
    img = lut[(np.clip(speed / vmax, 0, 1) * 255).astype(np.uint8)]
    _draw_colourbar(img, lut, vmax)
    if label:
        _draw_label(img, label)
    return img


def global_vmax(flow_ds, ks, step=None, percentile=99, n_sample=50):
    """
    Speed at ``percentile`` over a sample of pairs, used as the fixed colour/arrow limit for a video.
    With ``step``, uses block-mean speeds (what the quiver shows) instead of per-pixel speeds.
    """
    sample = ks[np.unique(np.linspace(0, len(ks) - 1, min(n_sample, len(ks))).astype(int))]
    speeds = []
    for k in sample:
        flow = flow_ds[k]
        u, v = block_mean(flow, step)[2:] if step else (flow[..., 0], flow[..., 1])
        speeds.append(np.hypot(u, v).ravel())
    vmax = float(np.nanpercentile(np.concatenate(speeds), percentile))
    return vmax if vmax > 0 else 1.0


def make_flow_video(
    flow_path,
    out=None,
    *,
    kind="quiver",
    grid=32,
    vmax=None,
    arrow_scale=None,
    fps=10,
    dim=0.55,
    cmap="turbo",
    quality=6,
    source=None,
    progress=True,
) -> Path:
    """
    Render a flow file as an MP4.

    Parameters
    ----------
    flow_path : str or pathlib.Path
        Flow file written by ``compute_flow``.
    out : str or pathlib.Path, optional
        Output video. Defaults to ``<flow stem>_<kind>.mp4`` next to the flow file.
    kind : {"quiver", "speed"}
        "quiver": the first frame of each pair, dimmed, with block-mean arrows coloured by speed.
        "speed": a |flow| heatmap.
    grid : int
        Approximate number of arrows across the larger image dimension (quiver only).
    vmax : float, optional
        Speed (px/pair) at the top of the colour scale, fixed for the whole video. Defaults to the
        99th percentile over a sample of pairs.
    arrow_scale : float, optional
        Arrow length in px per px/pair (quiver only). Defaults to making a ``vmax`` arrow 0.9 of a
        grid cell, fixed for the whole video.
    fps : float
        Output frames per second (one video frame per flow pair).
    dim : float
        Background brightness factor (quiver only).
    cmap : str
        Matplotlib colormap name for speed.
    quality : int
        FFMPEG quality passed to imageio.
    source : str or pathlib.Path, optional
        Source file, if it has moved since the flow was computed (e.g. computed on another computer).

    Returns
    -------
    pathlib.Path
        The video file.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    flow_path = Path(flow_path)
    out = Path(out) if out else flow_path.with_name(f"{flow_path.stem}_{kind}.mp4")
    lut = colour_lut(cmap)

    with open_flow(flow_path) as f:
        config = read_config(f)
        flow_ds = f["flow"]
        ks = np.flatnonzero(f["done"][:])
        if len(ks) == 0:
            raise ValueError(f"{flow_path} has no computed pairs")
        i0, i1, t0 = f["i0"][:], f["i1"][:], f["t0"][:]
        H, W = flow_ds.shape[1:3]
        step = max(max(H, W) // grid, 1)

        if vmax is None:
            vmax = global_vmax(flow_ds, ks, step if kind == "quiver" else None)
        if arrow_scale is None:
            arrow_scale = 0.9 * step / vmax
        scaler = FrameScaler(float(f.attrs["lo"]), float(f.attrs["hi"]), "uint8")

        with (
            h5py.File(source or config["source"], "r") as src,
            imageio.get_writer(out, format="FFMPEG", fps=fps, codec="libx264", quality=quality) as writer,
        ):
            frames = src[config["frames"]]
            for k in tqdm(ks, desc=out.stem, unit="frame", disable=not progress):
                label = _time_label(t0[k], i0[k], i1[k])
                if kind == "quiver":
                    img = draw_quiver(scaler(frames[i0[k]]), flow_ds[k], step, arrow_scale, vmax, lut, label, dim)
                else:
                    img = draw_speed(flow_ds[k], vmax, lut, label)
                writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out
