"""
Score every run of a sweep folder from its saved flow files (TV-L1 is not rerun).

Real-data scores work on any source. They split the frame into structured regions (strong
gradients: the lanes and their edges), stars, and diffuse glow, because frame-wide averages are
dominated by featureless glow and noise:

    warp_lane, warp_diffuse    mean |I0 - I1 warped back|; compare with noise_floor (two noisy frames
                               can't match better, so values below it mean the flow is fitting noise)
    speed_lane_p50/_p95        flow speed on the structured regions, px/pair
    speed_diffuse_p50          median speed in diffuse glow, where there is little to track
    temporal_change_lane       median |flow_k+1 - flow_k| on structure, relative to its speed
                               (only for contiguous pairs: stride == gap)
    rough_diffuse              mean |flow gradient| in diffuse glow
    star_over_ring             median speed on stars / around them (<1: static stars pin the flow)

Truth scores are added when the source is a benchmark (``flow.benchmark``), per pair against
``true_flow``:

    epe_lane, epe_lane_p95     endpoint error |est - true| on moving lanes, px
    speed_recovery             median |est| / |true| on moving lanes (1 = full speed recovered)
    vorticity_corr, shear_corr correlation of the smoothed flow's vorticity / shear with the truth
    false_still, false_bright  median speed in still glow / in patches that only change brightness
    epe_<case>                 epe_lane per motion case
    drift_rank, structure_rank, combined_rank   ranks by epe_lane, by mean structure correlation,
                               and their average (1 = best)
"""

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import cv2

from auroral_processing.flow.compute_flow import FrameScaler
from auroral_processing.flow.io import open_flow, read_config
from auroral_processing.flow.params import TVL1Params, external_name
from auroral_processing.flow.benchmark import find_stars, measure_noise

BORDER = 16  # px ignored at the frame edge, where the flow is unreliable

# (column, title, lower_is_better) for the heatmaps
REAL_PANELS = [
    ("warp_diffuse", "Warp residual, diffuse glow", True),
    ("speed_diffuse_p50", "Median speed in diffuse glow, px/pair", True),
    ("speed_lane_p95", "p95 speed on structure, px/pair", None),
    ("temporal_change_lane", "Pair-to-pair change on structure", True),
    ("rough_diffuse", "Flow roughness in diffuse glow", True),
    ("star_over_ring", "Speed on stars / around them", None),
]
TRUTH_PANELS = [
    ("epe_lane", "Error on moving lanes, px", True),
    ("epe_lane_p95", "p95 error on moving lanes, px", True),
    ("speed_recovery", "Speed recovered (1 = all)", None),
    ("structure_corr", "Vorticity/shear correlation", False),
    ("false_still", "False motion in still glow, px", True),
    ("false_bright", "False motion in brightening glow, px", True),
    ("combined_rank", "Combined rank (drift + structure)", True),
]


def structured_mask(img, frac=0.15):
    """Strong gradients of the smoothed frame: the lanes and their edges."""
    g = cv2.GaussianBlur(img, (0, 0), 3)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m > np.percentile(m, 100 * (1 - frac))


def _derivatives(flow, sigma=4):
    u, v = (cv2.GaussianBlur(flow[..., c], (0, 0), sigma) for c in range(2))
    ux, uy = np.gradient(u, axis=1), np.gradient(u, axis=0)
    vx, vy = np.gradient(v, axis=1), np.gradient(v, axis=0)
    return vx - uy, np.hypot(ux - vy, uy + vx)  # vorticity, shear magnitude


def _corr(a, b):
    if a.std() < 1e-4 or b.std() < 1e-4:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


class _Inputs:
    """Frames, masks and truth shared by all runs of a sweep (they all use the same pairs)."""

    def __init__(self, first_run, source=None):
        with open_flow(first_run) as f:
            cfg = read_config(f)
            self.i0, self.i1 = f["i0"][:], f["i1"][:]
            self.scaler = FrameScaler(float(f.attrs["lo"]), float(f.attrs["hi"]))
        self.contiguous = cfg["stride"] == cfg["gap"]
        self.source = Path(source or cfg["source"])
        sc = self.scaler
        with h5py.File(self.source, "r") as f:
            ds = f[cfg["frames"]]
            self.A = np.stack([sc(ds[int(i)]) for i in self.i0])
            self.B = np.stack([sc(ds[int(i)]) for i in self.i1])
            self.truth = "true_flow" in f
            if self.truth:
                if not ((self.i1 == self.i0 + 1) & (self.i0 % 2 == 0)).all():
                    raise ValueError("benchmark sweeps must use --gap 1 --stride 2 so each pair is (a, b)")
                p = (self.i0 // 2).tolist()
                self.true = f["true_flow"][p]
                self.case = f["case"].asstr()[p]
                self.lanes = f["masks/lanes"][p]
                self.bright = f["masks/bright_patch"][p]
                self.stars = f["masks/stars"][:]
            else:
                self.stars = find_stars(ds[np.unique(np.linspace(0, ds.shape[0] - 1, 50).astype(int)).tolist()]
                                        .astype(np.float32).mean(0), 0.08 * (sc.hi - sc.lo))
            n = min(ds.shape[0] - 2, 5)
            sigma, _ = measure_noise(ds, np.unique(np.linspace(0, ds.shape[0] - 2, n).astype(int)), self.stars)
        self.noise_floor = 1.128 * sigma / (sc.hi - sc.lo)  # E|n0 - n1| for Gaussian noise, scaled units
        H, W = self.A.shape[1:]
        self.edge = np.zeros((H, W), bool)
        self.edge[BORDER:-BORDER, BORDER:-BORDER] = True
        self.L = np.stack([structured_mask(a) & ~self.stars for a in self.A])
        self.ring = cv2.dilate(self.stars.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~self.stars
        self.grid = np.mgrid[:H, :W].astype(np.float32)[::-1]  # (x, y)


def _real_scores(F, I: _Inputs):
    s = np.hypot(F[..., 0], F[..., 1])
    L, diffuse = I.L & I.edge, ~I.L & ~I.stars & I.edge
    xx, yy = I.grid
    warp_lane, warp_diffuse = [], []
    for k in range(len(F)):
        w = cv2.remap(I.B[k], xx + F[k, ..., 0], yy + F[k, ..., 1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        e = np.abs(w - I.A[k])
        warp_lane.append(np.nanmean(e[L[k]]))
        warp_diffuse.append(np.nanmean(e[diffuse[k]]))
    gx = np.abs(np.diff(F, axis=2)).sum(-1)[:, :-1, :]
    gy = np.abs(np.diff(F, axis=1)).sum(-1)[:, :, :-1]
    rough = gx + gy
    row = dict(
        noise_floor=I.noise_floor,
        warp_lane=float(np.mean(warp_lane)),
        warp_diffuse=float(np.mean(warp_diffuse)),
        speed_lane_p50=float(np.median(s[L])),
        speed_lane_p95=float(np.percentile(s[L], 95)),
        speed_diffuse_p50=float(np.median(s[diffuse])),
        rough_diffuse=float(rough[diffuse[:, :-1, :-1]].mean()),
        star_over_ring=float(np.median(s[:, I.stars & I.edge]) / np.median(s[:, I.ring & I.edge]))
        if (I.stars & I.edge).any() else np.nan,
    )
    if I.contiguous and len(F) > 1:
        both = L[1:] & L[:-1]
        dt = np.hypot(*(F[1:] - F[:-1]).transpose(3, 0, 1, 2))
        row["temporal_change_lane"] = float(np.median(dt[both]) / np.median(s[L]))
    return row


def _truth_scores(F, I: _Inputs):
    T = I.true
    st = np.hypot(T[..., 0], T[..., 1])
    epe = np.hypot(*(F - T).transpose(3, 0, 1, 2))
    se = np.hypot(F[..., 0], F[..., 1])
    moving = I.lanes & (st > 1) & I.edge
    fast = moving & (st > 2)
    still = (st < 0.2) & ~I.bright & ~I.stars & I.edge
    vort, shear = [], []
    for k in range(len(F)):
        region = cv2.dilate((st[k] > 1).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & I.edge
        if region.sum() < 100:
            continue
        (ve, se_), (vt, sht) = _derivatives(F[k]), _derivatives(T[k])
        vort.append(_corr(ve[region], vt[region]))
        shear.append(_corr(se_[region], sht[region]))
    row = dict(
        epe_lane=float(epe[moving].mean()),
        epe_lane_p95=float(np.percentile(epe[moving], 95)),
        speed_recovery=float(np.median(se[fast] / st[fast])),
        vorticity_corr=float(np.nanmean(vort)),
        shear_corr=float(np.nanmean(shear)),
        false_still=float(np.median(se[still])),
        false_bright=float(np.median(se[I.bright & I.edge])) if (I.bright & I.edge).any() else np.nan,
    )
    row["structure_corr"] = (row["vorticity_corr"] + row["shear_corr"]) / 2
    for case in dict.fromkeys(I.case):
        m = moving & (I.case == case)[:, None, None]
        row[f"epe_{case}"] = float(epe[m].mean()) if m.any() else np.nan
    return row


def _runs(sweep_dir):
    """(run name, flow path, {setting: value}) for each run, in summary.csv order when there is one."""
    summary = sweep_dir / "summary.csv"
    names = list(pd.read_csv(summary)["run"]) if summary.exists() else sorted(p.parent.name for p in sweep_dir.glob("*/flow.h5"))
    out = []
    for name in names:
        path = sweep_dir / name / "flow.h5"
        if not path.exists():
            continue
        with open_flow(path) as f:
            out.append((name, path, read_config(f)["tvl1"]))
    if not out:
        raise FileNotFoundError(f"no <run>/flow.h5 in {sweep_dir}")
    return out


def score_sweep(sweep_dir, *, source=None, plot=True, progress=True) -> Path:
    """
    Score each run in ``sweep_dir`` and write ``scores.csv`` (and ``scores_heatmaps.png``).

    Parameters
    ----------
    sweep_dir : str or pathlib.Path
        Folder from ``run_sweep`` / ``aurora flow sweep``.
    source : str or pathlib.Path, optional
        Source file, if it has moved since the sweep (e.g. the sweep ran on another computer).
    plot : bool
        Also draw the heatmaps of the main scores.

    Returns
    -------
    pathlib.Path
        The scores CSV.
    """
    sweep_dir = Path(sweep_dir)
    runs = _runs(sweep_dir)
    I = _Inputs(runs[0][1], source)
    rows = []
    for n, (name, path, tvl1) in enumerate(runs, 1):
        with open_flow(path) as f:
            if not (np.array_equal(f["i0"][:], I.i0) and f.attrs["complete"]):
                print(f"skipping {name}: incomplete or different pairs")
                continue
            F = f["flow"][:]
        row = {"run": name, **tvl1, **_real_scores(F, I)}
        if I.truth:
            row.update(_truth_scores(F, I))
        rows.append(row)
        if progress:
            print(f"[{n}/{len(runs)}] {name}")
    df = pd.DataFrame(rows)
    if I.truth:
        df["drift_rank"] = df["epe_lane"].rank()
        df["structure_rank"] = df["structure_corr"].rank(ascending=False)
        df["combined_rank"] = (df["drift_rank"] + df["structure_rank"]) / 2
    out = sweep_dir / "scores.csv"
    df.to_csv(out, index=False)
    if plot:
        plot_scores(df, sweep_dir / "scores_heatmaps.png", title=sweep_dir.name)
    return out


def varied_settings(df) -> list[str]:
    names = [external_name(n) for n in TVL1Params.__dataclass_fields__]
    return [n for n in names if n in df and df[n].nunique() > 1]


def plot_scores(df, out, title=""):
    """
    One panel per main score: a heatmap when exactly two settings vary, otherwise bars sorted by
    the score. The best run in each panel is outlined (for scores with a clear direction).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels = [p for p in (TRUTH_PANELS if "epe_lane" in df else REAL_PANELS) if p[0] in df]
    varied = varied_settings(df)
    cols = 3 if len(panels) > 4 else len(panels)
    rows = -(-len(panels) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), layout="constrained", squeeze=False)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")

    for ax, (col, label, lower) in zip(axs.flat, panels):
        hint = {True: " (lower better)", False: " (higher better)", None: ""}[lower]
        ax.set_title(label + hint, fontsize=10)
        if len(varied) == 2:
            t = df.pivot_table(index=varied[0], columns=varied[1], values=col)
            vals = t.values
            im = ax.imshow(vals, cmap="viridis" if lower is not True else "viridis_r", aspect="auto")
            ax.set_xticks(range(t.shape[1]), [f"{v:g}" for v in t.columns])
            ax.set_yticks(range(t.shape[0]), [f"{v:g}" for v in t.index])
            ax.set_xlabel(varied[1])
            ax.set_ylabel(varied[0])
            span = np.nanmax(vals) - np.nanmin(vals) or 1
            for (i, j), v in np.ndenumerate(vals):
                dark = (v - np.nanmin(vals)) / span < 0.5
                if lower is True:
                    dark = not dark
                ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=8, color="white" if dark else "black")
            if lower is not None and np.isfinite(vals).any():
                i, j = np.unravel_index(np.nanargmin(vals) if lower else np.nanargmax(vals), vals.shape)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2))
            fig.colorbar(im, ax=ax, shrink=0.8)
        else:
            d = df.sort_values(col, ascending=lower is not False)
            ax.barh(d["run"], d[col], color="#4c78a8")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=7)
            if col == "warp_diffuse" and "noise_floor" in df:
                ax.axvline(df["noise_floor"].iloc[0], color="k", ls="--", lw=1)
    if "warp_diffuse" in [p[0] for p in panels] and "noise_floor" in df:
        title += f"   (noise floor for warp residual ≈ {df['noise_floor'].iloc[0]:.3f})"
    fig.suptitle(title)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
