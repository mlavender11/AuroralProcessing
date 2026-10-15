"""
Dense optical flow (Farneback + TV-L1) on a video or an HDF5 image stack,
rendered as an overlay movie.

Unlike correlation PIV, these solve for a per-pixel displacement field and
tolerate brightness change -- which is why they suit aurora better.

    pip install opencv-contrib-python numpy matplotlib h5py   # contrib needed for TV-L1

Video input:
    python optical_flow.py video.mp4 --gap 5 --stop 200
    python optical_flow.py video.mp4 --method tvl1 --grid 24 --arrow-scale 3

HDF5 input (datasets named by their path inside the file):
    python optical_flow.py data.h5 --images /camera/frames --time /camera/time
    python optical_flow.py data.h5 --images imgs --time t_us --time-scale 1e-6
    python optical_flow.py data.h5 --images imgs --fps 30       # no time dataset
    python optical_flow.py data.h5 --list                       # show datasets and exit

    The image dataset is (n_frames, ny, nx) or (n_frames, ny, nx, 3/4) by default;
    use --frame-axis if time is on another axis. The time dataset must have one
    entry per frame: numbers (seconds, or scaled by --time-scale) or datetimes /
    ISO strings. ISO strings may carry Z, UTC or +hh:mm offsets (converted to
    UTC); strings without one are taken as UTC. Unix-style epoch numbers work
    as-is; add --unix-time UT1 (or UTC, ...) to label frames with clock time.

        python optical_flow.py asi.h5 --images imgs --time ut1_unix --unix-time UT1
    dt is taken from the actual timestamps of each pair, so an
    irregular cadence is handled correctly. Non-uint8 images are stretched to
    8 bit with one global percentile window (--clip) so brightness changes
    between frames are preserved.

Outputs per method, in --outdir:
    {method}_overlay.mp4    original frames with flow arrows drawn on top
    {method}_series.npz     x, y, t, dt, U, V   (U,V in px/s, shape (n_fields, ny, nx))
    {method}_hsv.mp4        colour-wheel view of the dense field   (--hsv)
    mean_flow.png           time-averaged field
"""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np


H5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".he5", ".nc"}


# ----------------------------------------------------------------- frame sources

class VideoSource:
    """Frames from anything cv2.VideoCapture can open; time = index / fps."""

    def __init__(self, path):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n = n if n > 0 else 10 ** 9          # some containers don't report it
        cap.release()

    def describe(self):
        return f"video {self.path}: {self.n} frames @ {self.fps:.3f} fps"

    def time(self, i):
        return i / self.fps

    def utc(self, i):
        return None

    def frames(self, indices):
        """Yield (idx, gray uint8) for a sorted iterable of indices (sequential read)."""
        indices = list(indices)
        if not indices:
            return
        want, last = set(indices), indices[-1]
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        i = indices[0]
        while i <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                yield i, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            i += 1
        cap.release()


def list_h5(f):
    rows = []
    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            rows.append(f"  /{name:40s} {str(obj.shape):22s} {obj.dtype}")
    f.visititems(visit)
    return "\n".join(rows) or "  (no datasets)"


_TZ_RE = None


def _iso_to_utc(s):
    """ISO-8601 string (with or without Z / +hh:mm offset) -> naive UTC datetime64[ns].

    The offset is stripped and applied by hand so numpy never sees it (it
    warns or fails on time-zone suffixes) and nanosecond fractions survive.
    """
    global _TZ_RE
    import re
    if _TZ_RE is None:
        _TZ_RE = re.compile(r"(?:\s*(?:UTC|GMT|Z)|([+-])(\d{2}):?(\d{2})?)$", re.I)
    s = s.strip().replace(" ", "T", 1) if "T" not in s else s.strip()
    off = np.timedelta64(0, "ns")
    m = _TZ_RE.search(s) if len(s) > 10 else None     # don't eat a bare date's "-dd"
    if m:
        if m.group(1):
            mins = int(m.group(2)) * 60 + int(m.group(3) or 0)
            off = np.timedelta64(mins * 60 * 10**9, "ns") * (1 if m.group(1) == "+" else -1)
        s = s[:m.start()]
    return np.datetime64(s, "ns") - off              # local - offset = UTC


def parse_times(raw, scale, unix=False):
    """Time dataset -> (float seconds since the first frame, absolute UTC datetime64 or None).

    Accepted: ISO strings (naive = UTC, or with Z / offsets), datetime64,
    unix epoch numbers (UTC) with unix=True, or any other numeric seconds.
    """
    raw = np.asarray(raw).ravel()
    if raw.dtype.kind in "SOU":                        # ISO strings / bytes
        s = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        raw = np.array([_iso_to_utc(x) for x in s], dtype="datetime64[ns]")
    if raw.dtype.kind == "M":                          # datetime64 (taken as UTC)
        raw = raw.astype("datetime64[ns]")
        rel = (raw - raw[0]).astype(np.int64).astype(np.float64) * 1e-9
        return rel, raw[0]
    sec = raw.astype(np.float64) * scale
    if unix:
        t0 = np.datetime64(int(round(sec[0] * 1e9)), "ns")
        return sec - sec[0], t0
    return sec - sec[0], None


class H5Source:
    """Frames from an image dataset in an HDF5 file, times from a time dataset."""

    def __init__(self, path, images, time=None, time_scale=1.0, fps=None,
                 frame_axis=0, clip=(0.5, 99.5), start=0, stop=None, unix=False):
        try:
            import h5py
        except ImportError:
            raise SystemExit("HDF5 input needs h5py (pip install h5py)")
        self.path = path
        self.f = h5py.File(path, "r")

        if not images:
            raise SystemExit(f"--images is required for HDF5 input. Datasets in {path}:\n"
                             + list_h5(self.f))
        self.ds = self._dataset(images)
        self.axis = frame_axis % self.ds.ndim
        if self.ds.ndim not in (3, 4):
            raise SystemExit(f"image dataset {images} has shape {self.ds.shape}; "
                             "expected (n, ny, nx) or (n, ny, nx, channels)")
        self.n = self.ds.shape[self.axis]

        # pipeline outputs record their time dataset (and its scale) in attributes
        if not time and "time_dataset" in self.ds.attrs:
            time = self.ds.attrs["time_dataset"]
            time = time.decode() if isinstance(time, bytes) else str(time)
            if time not in self.f:
                time = None
            else:
                print(f"using time dataset /{time} (from {images} attrs)")
        if time and not unix and "time_scale" in self.f[time].attrs:
            unix = self.f[time].attrs["time_scale"]
            unix = unix.decode() if isinstance(unix, bytes) else str(unix)

        self.t_utc0 = None                   # absolute time of frame 0, if known
        self.scale_name = unix or "UTC"      # what the clock labels are called
        if time:
            tds = self._dataset(time)
            self.t, self.t_utc0 = parse_times(tds[()], time_scale, bool(unix))
            if len(self.t) != self.n:
                raise SystemExit(f"time dataset {time} has {len(self.t)} entries "
                                 f"but {images} has {self.n} frames on axis {self.axis}")
        elif fps:
            self.t = np.arange(self.n) / fps
        else:
            raise SystemExit("HDF5 input needs --time <dataset> or --fps")

        d = np.diff(self.t)
        if np.any(d <= 0):
            print(f"warning: time is not strictly increasing at {np.sum(d <= 0)} place(s); "
                  "those pairs will be skipped")
        good = d[d > 0]
        self.fps = 1.0 / np.median(good) if good.size else 1.0
        self.irregular = good.size and np.std(good) / np.median(good) > 0.01

        # global 8-bit stretch window, from a sample of frames in the range used
        self.lo = self.hi = None
        if self.ds.dtype != np.uint8:
            stop = self.n if stop is None else min(stop, self.n)
            sample = np.unique(np.linspace(start, max(stop - 1, start), 20).astype(int))
            vals = np.concatenate([self._raw_gray(i)[::4, ::4].ravel() for i in sample])
            vals = vals[np.isfinite(vals)]
            self.lo, self.hi = np.percentile(vals, clip)
            if self.hi <= self.lo:
                self.hi = self.lo + 1.0

    def _dataset(self, name):
        try:
            return self.f[name]
        except KeyError:
            raise SystemExit(f"no dataset '{name}' in {self.path}. Datasets:\n"
                             + list_h5(self.f))

    def _raw_gray(self, i):
        sl = [slice(None)] * self.ds.ndim
        sl[self.axis] = i
        fr = np.asarray(self.ds[tuple(sl)], dtype=np.float32)
        if fr.ndim == 3:                                   # colour: (ny, nx, c)
            if fr.shape[-1] in (3, 4):
                fr = fr[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
            else:
                fr = fr.mean(-1)
        return fr

    def describe(self):
        s = (f"hdf5 {self.path}: {self.ds.name} {self.ds.shape} {self.ds.dtype}, "
             f"{self.n} frames, median cadence {1/self.fps*1e3:.2f} ms ({self.fps:.3f} fps)")
        if self.irregular:
            s += "  [irregular cadence -- per-pair dt used]"
        if self.t_utc0 is not None:
            s += f"\n  {self.scale_name} {self.utc(0)} .. {self.utc(self.n - 1)}"
        if self.lo is not None:
            s += f"\n  8-bit stretch window [{self.lo:.4g}, {self.hi:.4g}]"
        return s

    def time(self, i):
        return float(self.t[i])

    def utc(self, i):
        """ISO clock-time string (in self.scale_name) for frame i, or None."""
        if self.t_utc0 is None:
            return None
        ts = self.t_utc0 + np.timedelta64(int(round(self.t[i] * 1e9)), "ns")
        return str(ts.astype("datetime64[ms]"))

    def frames(self, indices):
        for i in indices:
            if i >= self.n:
                break
            fr = self._raw_gray(i)
            if self.lo is None:
                g = fr.astype(np.uint8)
            else:
                fr = np.nan_to_num(fr, nan=self.lo)
                g = np.clip((fr - self.lo) / (self.hi - self.lo) * 255, 0, 255).astype(np.uint8)
            yield i, g


# ----------------------------------------------------------------- flow methods

TVL1_PARAMS = {
    # cli name      setter                 type   default  help
    "lambda":      ("setLambda",           float, 0.15,  "data weight: higher = follow the images more, smooth less"),
    "tau":         ("setTau",              float, 0.25,  "time step of the solver (keep <= 0.25 for stability)"),
    "theta":       ("setTheta",            float, 0.3,   "coupling between data and smoothness terms"),
    "scales":      ("setScalesNumber",     int,   5,     "pyramid levels: more = can follow larger motions"),
    "scale-step":  ("setScaleStep",        float, 0.8,   "size ratio between pyramid levels"),
    "warps":       ("setWarpingsNumber",   int,   5,     "warps per level: more = better for large/nonlinear motion"),
    "epsilon":     ("setEpsilon",          float, 0.01,  "convergence threshold (smaller = more accurate, slower)"),
    "inner-iter":  ("setInnerIterations",  int,   30,    "inner iterations per warp"),
    "outer-iter":  ("setOuterIterations",  int,   10,    "outer iterations per warp"),
    "gamma":       ("setGamma",            float, 0.0,   "weight of brightness-change term (>0 tolerates illumination change)"),
    "median":      ("setMedianFiltering",  int,   5,     "median filter size on the flow (1 = off; odd)"),
}


def make_tvl1(args=None):
    if not hasattr(cv2, "optflow"):
        raise SystemExit("TV-L1 needs opencv-contrib-python (pip install opencv-contrib-python)")
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    used = {}
    for name, (setter, _, default, _) in TVL1_PARAMS.items():
        val = getattr(args, "tvl1_" + name.replace("-", "_"), None) if args else None
        val = default if val is None else val
        getattr(tvl1, setter)(val)
        used[name] = val
    return tvl1, used


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
    p.add_argument("source", help="video file or HDF5 file")
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

    h = p.add_argument_group("HDF5 input")
    h.add_argument("--images", help="image dataset name/path, e.g. /camera/frames")
    h.add_argument("--time", help="time dataset name/path (one entry per frame)")
    h.add_argument("--time-scale", type=float, default=1.0,
                   help="multiply numeric times by this to get seconds (e.g. 1e-6 for us)")
    h.add_argument("--unix-time", nargs="?", const="UTC", default=None, metavar="SCALE",
                   help="numeric times are unix-style epoch seconds; label frames with the "
                        "clock time in that scale (e.g. --unix-time UT1; default UTC)")
    h.add_argument("--fps", type=float, default=None,
                   help="frame rate to assume when there is no time dataset")
    h.add_argument("--frame-axis", type=int, default=0,
                   help="axis of the image dataset that indexes frames")
    h.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5), metavar=("PLO", "PHI"),
                   help="percentiles for the global 8-bit stretch of non-uint8 images")
    h.add_argument("--list", action="store_true", help="list the HDF5 datasets and exit")
    h.add_argument("--hdf5", action="store_true",
                   help="treat the source as HDF5 regardless of its extension")

    g = p.add_argument_group("TV-L1 parameters (OpenCV defaults shown)")
    for name, (_, typ, default, desc) in TVL1_PARAMS.items():
        g.add_argument(f"--tvl1-{name}", type=typ, default=None,
                       help=f"{desc} [default {default}]")
    args = p.parse_args()

    is_h5 = args.hdf5 or args.images or Path(args.source).suffix.lower() in H5_SUFFIXES
    if args.list:
        import h5py
        with h5py.File(args.source, "r") as f:
            print(list_h5(f))
        return

    if is_h5:
        src = H5Source(args.source, args.images, args.time, args.time_scale, args.fps,
                       args.frame_axis, tuple(args.clip), args.start, args.stop,
                       args.unix_time)
    else:
        src = VideoSource(args.source)

    stride = args.stride or args.gap
    out = Path(args.outdir); out.mkdir(exist_ok=True)
    stop = src.n if args.stop is None else min(args.stop, src.n)
    t0 = src.time(args.start)
    print(src.describe())
    print(f"gap={args.gap} frames -> nominal dt={args.gap / src.fps * 1e3:.2f} ms")

    methods = {}
    if args.method in ("farneback", "both"):
        methods["farneback"] = farneback
    if args.method in ("tvl1", "both"):
        tvl1, tvl1_used = make_tvl1(args)
        methods["tvl1"] = lambda a, b: tvl1.calc(a, b, None)
        print("tvl1: " + "  ".join(f"{k}={v}" for k, v in tvl1_used.items()))

    # ---------------- pass 1: compute the flow fields ----------------
    buf = deque(maxlen=args.gap + 1)
    results = {k: {"U": [], "V": [], "t": [], "dt": [], "idx": []} for k in methods}
    hsv_frames = {k: [] for k in methods} if args.hsv else None
    X = Y = None
    skipped = 0

    for idx, gray in src.frames(range(args.start, stop)):
        buf.append((idx, gray))
        if len(buf) == args.gap + 1 and (idx - args.start - args.gap) % stride == 0:
            (ia, a), (ib, b) = buf[0], buf[-1]
            dt = src.time(ib) - src.time(ia)          # real elapsed time for this pair
            if not dt > 0:
                skipped += 1
                continue
            for name, fn in methods.items():
                flow_px = fn(a, b)                    # px per pair
                X, Y, u, v = downsample(flow_px / dt, args.grid)   # px/s
                results[name]["U"].append(u)
                results[name]["V"].append(v)
                results[name]["t"].append(src.time(ia) - t0)
                results[name]["dt"].append(dt)
                results[name]["idx"].append(ia)
                if args.hsv:
                    hsv_frames[name].append(flow_px)
            if (idx - args.start) % (stride * 20) == 0:
                print(f"  computing... frame {idx}")

    if X is None:
        raise SystemExit("no pairs produced -- check --start/--stop/--gap")
    if skipped:
        print(f"skipped {skipped} pair(s) with non-positive dt")

    for name in results:
        results[name]["U"] = np.stack(results[name]["U"])
        results[name]["V"] = np.stack(results[name]["V"])

    # ---------------- pass 2: render the overlay movies ----------------
    lut = colour_lut(args.cmap)
    spacing = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 16.0
    out_fps = args.out_fps or max(src.fps / stride, 1.0)

    for name, r in results.items():
        U, V = r["U"], r["V"]
        mag = np.hypot(U, V)
        vmax = args.vmax or float(np.nanpercentile(mag, 98)) or 1.0
        # arrow length: p95 speed should span ~0.9 of the grid spacing
        px_scale = args.arrow_scale or (0.9 * spacing / max(np.nanpercentile(mag, 95), 1e-6))

        writer, wrote = None, 0
        want = dict(zip(r["idx"], range(len(r["idx"]))))
        for fidx, gray in src.frames(sorted(want)):
            k = want[fidx]
            u = src.utc(fidx)
            when = f"{u[11:]} {src.scale_name}" if u else f"t={r['t'][k]:8.3f}s"
            img = draw_overlay(
                gray, X, Y, U[k], V[k], px_scale, vmax, lut,
                label=f"{name}  {when}  gap={args.gap}f  dt={r['dt'][k]*1e3:.1f}ms",
                dim=args.dim)
            if writer is None:
                h_, w_ = img.shape[:2]
                writer = cv2.VideoWriter(str(out / f"{name}_overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         out_fps, (w_, h_))
            writer.write(img)
            wrote += 1
        if writer is not None:
            writer.release()

        if args.hsv:
            hm = float(np.nanpercentile([np.hypot(f[..., 0], f[..., 1]).max()
                                         for f in hsv_frames[name]], 90)) or 1.0
            hw = None
            for f in hsv_frames[name]:
                im = flow_to_hsv(f, hm)
                if hw is None:
                    h_, w_ = im.shape[:2]
                    hw = cv2.VideoWriter(str(out / f"{name}_hsv.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w_, h_))
                hw.write(im)
            if hw is not None:
                hw.release()

        np.savez_compressed(out / f"{name}_series.npz",
                            x=X, y=Y, t=np.array(r["t"]), dt=np.array(r["dt"]),
                            idx=np.array(r["idx"]), U=U, V=V, t0=t0,
                            clock=np.array([src.utc(i) or "" for i in r["idx"]]),
                            clock_scale=getattr(src, "scale_name", ""),
                            params=repr(tvl1_used if name == "tvl1" else {}),
                            fps=src.fps, gap=args.gap, stride=stride,
                            source=str(args.source))

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
