"""
Dense optical flow (Farneback + TV-L1) on a video, rendered as an overlay movie.

Unlike correlation PIV, these solve for a per-pixel displacement field and
tolerate brightness change -- which is why they suit aurora better.

    pip install opencv-contrib-python numpy matplotlib     # contrib needed for TV-L1

    python optical_flow.py video.mp4 --gap 5 --stop 200
    python optical_flow.py video.mp4 --method tvl1 --grid 24 --arrow-scale 3
    python optical_flow.py video.mp4 --hsv                 # also write the colour-wheel movie

Outputs per method, in --outdir:
    {method}_overlay.mp4    original frames with flow arrows drawn on top
    {method}_series.npz     x, y, t, U, V   (U,V in px/s, shape (n_fields, ny, nx))
    {method}_hsv.mp4        colour-wheel view of the dense field   (--hsv)
    mean_flow.png           time-averaged field
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------- flow methods

def make_tvl1():
    if not hasattr(cv2, "optflow"):
        raise SystemExit("TV-L1 needs opencv-contrib-python (pip install opencv-contrib-python)")
    return cv2.optflow.DualTVL1OpticalFlow_create()


def farneback(a, b):
    return cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=4, winsize=25, iterations=4,
        poly_n=7, poly_sigma=1.5, flags=0,
    )


# ----------------------------------------------------------------- rendering

def colour_lut(name="turbo"):
    """256-entry BGR lookup table from a matplotlib colormap."""
    import matplotlib
    cmap = matplotlib.colormaps[name]
    rgb = (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()          # RGB -> BGR


def flow_to_hsv(flow, max_mag):
    """Colour-wheel view: hue = direction, value = magnitude."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)          # 0..180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_overlay(frame_gray, X, Y, u, v, px_scale, vmax, lut,
                 label="", dim=0.55, thickness=1):
    """Original frame + one arrow per grid point, coloured by speed.

    u, v are in px/s; px_scale converts them to arrow length in pixels.
    """
    img = cv2.cvtColor((frame_gray * dim).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    speed = np.hypot(u, v)

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            if not np.isfinite(u[j, i]):
                continue
            x0, y0 = int(X[j, i]), int(Y[j, i])
            dx, dy = u[j, i] * px_scale, v[j, i] * px_scale
            if not np.isfinite(dx) or (abs(dx) < 0.5 and abs(dy) < 0.5):
                continue
            c = lut[int(np.clip(speed[j, i] / vmax, 0, 1) * 255)]
            cv2.arrowedLine(img, (x0, y0), (int(x0 + dx), int(y0 + dy)),
                            tuple(int(z) for z in c), thickness,
                            cv2.LINE_AA, tipLength=0.3)

    # colour scale strip, on a dark backing so it stays legible
    h, w = img.shape[:2]
    bar_w, bar_h, pad = min(180, w // 3), 8, 10
    cv2.rectangle(img, (0, h - 34), (bar_w + 2 * pad, h), (0, 0, 0), -1)
    if label:
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    ramp = lut[(np.linspace(0, 255, bar_w)).astype(int)][None].repeat(bar_h, 0)
    img[h-pad-bar_h:h-pad, pad:pad+bar_w] = ramp
    cv2.putText(img, "0", (pad, h - pad - bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{vmax:.0f} px/s", (pad + bar_w - 52, h - pad - bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (pad, pad + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def downsample(flow, n):
    """Average the dense field onto an n-ish grid -> X, Y, u, v (like a PIV field)."""
    h, w = flow.shape[:2]
    step = max(h // n, w // n, 1)
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    half = step // 2
    u = np.stack([[flow[max(y-half, 0):y+half+1, max(x-half, 0):x+half+1, 0].mean()
                   for x in xs] for y in ys])
    v = np.stack([[flow[max(y-half, 0):y+half+1, max(x-half, 0):x+half+1, 1].mean()
                   for x in xs] for y in ys])
    X, Y = np.meshgrid(xs, ys)
    return X, Y, u, v


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--method", choices=["farneback", "tvl1", "both"], default="both")
    p.add_argument("--gap", type=int, default=1, help="frames between the pair")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="frames between pairs (default = gap)")
    p.add_argument("--grid", type=int, default=32, help="arrows across the frame")
    p.add_argument("--arrow-scale", type=float, default=None,
                   help="arrow length multiplier (default: auto, ~0.9 grid spacing at p95 speed)")
    p.add_argument("--vmax", type=float, default=None,
                   help="speed [px/s] at the top of the colour scale (default: p98)")
    p.add_argument("--dim", type=float, default=0.55,
                   help="how much to dim the background frame, 0..1")
    p.add_argument("--out-fps", type=float, default=None,
                   help="output video fps (default: real time)")
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--hsv", action="store_true", help="also write the colour-wheel movie")
    p.add_argument("--outdir", default="flow_out")
    args = p.parse_args()

    stride = args.stride or args.gap
    out = Path(args.outdir); out.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = args.gap / fps
    print(f"{n_frames} frames @ {fps:.3f} fps   gap={args.gap} -> dt={dt*1e3:.2f} ms")

    methods = {}
    if args.method in ("farneback", "both"):
        methods["farneback"] = farneback
    if args.method in ("tvl1", "both"):
        tvl1 = make_tvl1()
        methods["tvl1"] = lambda a, b: tvl1.calc(a, b, None)

    # ---------------- pass 1: compute the flow fields ----------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    buf = []
    results = {k: {"U": [], "V": [], "t": [], "idx": []} for k in methods}
    hsv_frames = {k: [] for k in methods} if args.hsv else None
    X = Y = None
    idx = args.start

    while args.stop is None or idx < args.stop:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if len(buf) > args.gap + 1:
            buf.pop(0)

        if len(buf) == args.gap + 1 and (idx - args.start - args.gap) % stride == 0:
            a, b = buf[0], buf[-1]
            for name, fn in methods.items():
                flow = fn(a, b) / dt                      # px/s
                X, Y, u, v = downsample(flow, args.grid)
                results[name]["U"].append(u)
                results[name]["V"].append(v)
                results[name]["t"].append((idx - args.gap) / fps)
                results[name]["idx"].append(idx - args.gap)
                if args.hsv:
                    hsv_frames[name].append(flow * dt)    # px per frame-gap
            if (idx - args.start) % (stride * 20) == 0:
                print(f"  computing... frame {idx}")
        idx += 1
    cap.release()

    if X is None:
        raise SystemExit("no pairs produced -- check --start/--stop/--gap")

    for name in results:
        results[name]["U"] = np.stack(results[name]["U"])
        results[name]["V"] = np.stack(results[name]["V"])

    # ---------------- pass 2: render the overlay movies ----------------
    lut = colour_lut(args.cmap)
    spacing = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 16.0
    out_fps = args.out_fps or max(fps / stride, 1.0)

    for name, r in results.items():
        U, V = r["U"], r["V"]
        mag = np.hypot(U, V)
        vmax = args.vmax or float(np.nanpercentile(mag, 98)) or 1.0
        # arrow length: p95 speed should span ~0.9 of the grid spacing
        px_scale = args.arrow_scale or (0.9 * spacing / max(np.nanpercentile(mag, 95), 1e-6))

        cap = cv2.VideoCapture(args.video)
        writer, wrote = None, 0
        want = dict(zip(r["idx"], range(len(r["idx"]))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(r["idx"]))
        fidx = min(r["idx"])

        while fidx <= max(r["idx"]):
            ok, fr = cap.read()
            if not ok:
                break
            k = want.get(fidx)
            if k is not None:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                img = draw_overlay(
                    gray, X, Y, U[k], V[k], px_scale, vmax, lut,
                    label=f"{name}  t={r['t'][k]:6.2f}s  gap={args.gap}f",
                    dim=args.dim)
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(str(out / f"{name}_overlay.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             out_fps, (w, h))
                writer.write(img)
                wrote += 1
            fidx += 1
        cap.release()
        if writer is not None:
            writer.release()

        if args.hsv:
            hm = float(np.nanpercentile([np.hypot(f[..., 0], f[..., 1]).max()
                                         for f in hsv_frames[name]], 90)) or 1.0
            hw = None
            for f in hsv_frames[name]:
                im = flow_to_hsv(f, hm)
                if hw is None:
                    h, w = im.shape[:2]
                    hw = cv2.VideoWriter(str(out / f"{name}_hsv.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                hw.write(im)
            if hw is not None:
                hw.release()

        np.savez_compressed(out / f"{name}_series.npz",
                            x=X, y=Y, t=np.array(r["t"]), U=U, V=V,
                            fps=fps, dt=dt, gap=args.gap, stride=stride)

        print(f"{name:10s} {U.shape[0]:4d} fields  mean {mag.mean():6.2f} px/s  "
              f"vmax {vmax:6.2f}  arrow x{px_scale:.2f}  -> {name}_overlay.mp4 ({wrote} frames)")

    # ---------------- summary still ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 7), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        Um, Vm = r["U"].mean(0), r["V"].mean(0)
        speed = np.hypot(Um, Vm)
        im = ax.imshow(speed, origin="upper", cmap="viridis",
                       extent=[X.min(), X.max(), Y.max(), Y.min()])
        ax.quiver(X, Y, Um, -Vm, color="w")
        ax.set_title(f"{name}: mean of {r['U'].shape[0]} fields  [px/s]")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out / "mean_flow.png", dpi=130, bbox_inches="tight")
    print(f"wrote {out/'mean_flow.png'}")


if __name__ == "__main__":
    main()