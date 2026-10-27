"""
Flow estimators: TV-L1 (OpenCV), multi-pass PIV (OpenPIV) and FLCT (pyflct), behind one interface.

Every estimator returns a dense (H, W, 2) float32 flow in px/pair, u = +x right, v = +y down,
frame a -> frame b, so all methods write the same flow file. OpenPIV and pyflct are imported only
when their method is used.
"""

from __future__ import annotations
from contextlib import chdir
import shutil
import tempfile
import warnings
import numpy as np
import cv2

from auroral_processing.flow.params import TVL1Params, PIVParams, FLCTParams, make_tvl1


def _to_unit(img):
    """Image as float32 0-1, whichever dtype the scaler produced."""
    return img.astype(np.float32) / 255 if img.dtype == np.uint8 else img.astype(np.float32)


class TVL1Estimator:
    supports_init = True
    extra_metrics = ()

    def __init__(self, params: TVL1Params):
        self.tvl1 = make_tvl1(params)
        self.last = {}

    def calc(self, a, b, init=None):
        self.tvl1.setUseInitialFlow(init is not None)
        return self.tvl1.calc(a, b, init)


def pass_windows(window, first_window) -> tuple[int, ...]:
    """Window size of each PIV pass: halve from ``first_window`` while above ``window``, then ``window``."""
    sizes, w = [], int(first_window)
    while w > window:
        sizes.append(w)
        w //= 2
    return (*sizes, int(window))


class PIVEstimator:
    """
    ``openpiv.windef`` multi-pass PIV, the grid vectors bilinearly interpolated to every pixel.

    ``last["invalid_frac"]`` is the share of final-pass vectors that failed validation and were
    replaced by their neighbours' mean.
    """

    supports_init = False
    extra_metrics = ("invalid_frac",)

    def __init__(self, params: PIVParams):
        from openpiv import windef

        if not 0 <= params.overlap < 1:
            raise ValueError(f"PIV overlap must be in [0, 1), got {params.overlap}")
        if params.first_window < params.window:
            raise ValueError(f"PIV first_window ({params.first_window}) must be >= window ({params.window})")
        s = windef.PIVSettings()
        s.windowsizes = pass_windows(params.window, params.first_window)
        s.overlap = tuple(int(w * params.overlap) for w in s.windowsizes)
        s.num_iterations = len(s.windowsizes)
        s.min_max_u_disp = s.min_max_v_disp = (-params.max_disp, params.max_disp)
        s.sig2noise_threshold = params.s2n_threshold
        s.median_threshold = params.median_threshold
        s.normalized_correlation = bool(params.normalized)
        s.show_plot = s.show_all_plots = False
        self.settings = s
        self.last = {}

    def _grid(self, a, b):
        from openpiv import windef, validation, filters

        s = self.settings
        rows, cols = windef.get_field_shape(a.shape, s.windowsizes[0], s.overlap[0])
        if s.num_iterations > 1 and min(rows, cols) < 4:
            raise ValueError(f"PIV first-pass window {s.windowsizes[0]} px gives a {rows}x{cols} grid on a {a.shape} "
                             "frame; multi-pass needs at least 4x4. Use a smaller first_window.")
        x, y, u, v, s2n = windef.first_pass(a, b, s)
        u, v = np.ma.masked_array(u, mask=False), np.ma.masked_array(v, mask=False)
        flags = validation.typical_validation(u, v, s2n, s)
        u, v = filters.replace_outliers(u, v, flags, method=s.filter_method, max_iter=s.max_filter_iteration,
                                        kernel_size=s.filter_kernel_size)
        for i in range(1, s.num_iterations):
            x, y, u, v, _, flags = windef.multipass_img_deform(a, b, i, x, y, u, v, s)
        u, v = (np.ma.fix_invalid(c, fill_value=0.0).data for c in (u, v))
        return x, y, u, v, flags

    def calc(self, a, b, init=None):
        a, b = _to_unit(a), _to_unit(b)
        try:
            x, y, u, v, flags = self._grid(a, b)
        except ValueError as e:
            if "validation" not in str(e):
                raise
            warnings.warn(f"PIV found no valid vectors in this pair ({e}); writing zero flow")
            self.last = {"invalid_frac": 1.0}
            return np.zeros((*a.shape, 2), np.float32)
        self.last = {"invalid_frac": float(np.mean(flags))}
        # x, y are window centres on a regular grid (rows = image rows, v already +down)
        x0, dx = float(x[0, 0]), float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0
        y0, dy = float(y[0, 0]), float(y[1, 0] - y[0, 0]) if y.shape[0] > 1 else 1.0
        H, W = a.shape
        mx = np.broadcast_to(((np.arange(W, dtype=np.float32) - x0) / dx)[None, :], (H, W))
        my = np.broadcast_to(((np.arange(H, dtype=np.float32) - y0) / dy)[:, None], (H, W))
        mx, my = np.ascontiguousarray(mx), np.ascontiguousarray(my)
        dense = [cv2.remap(c.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                 for c in (u, v)]
        return np.dstack(dense)


class FLCTEstimator:
    """
    ``pyflct.flct`` with ``deltat = deltas = 1`` so velocities are px/pair.

    pyflct 0.3's default row order returns no y motion, so frames go in with ``order="column"``
    and the outputs are transposed back. That path round-trips the frames through a ``temp.dat`` in
    the working directory, so each estimator runs pyflct in its own temporary directory (parallel
    workers would otherwise delete each other's file). Pixels FLCT skips (below ``thresh``) get zero
    flow; ``last["invalid_frac"]`` is their share.
    """

    supports_init = False
    extra_metrics = ("invalid_frac",)

    def __init__(self, params: FLCTParams):
        import pyflct  # noqa: F401  (fail early if it isn't installed)

        self.p = params
        self.last = {}
        self.workdir = tempfile.mkdtemp(prefix="flct_")

    def __del__(self):
        shutil.rmtree(getattr(self, "workdir", ""), ignore_errors=True)

    def calc(self, a, b, init=None):
        import pyflct

        p = self.p
        skip = int(p.skip) if p.skip > 1 else None
        with chdir(self.workdir):
            vx, vy, vm = pyflct.flct(
                _to_unit(a).astype(np.float64), _to_unit(b).astype(np.float64), 1.0, 1.0, float(p.sigma),
                order="column", quiet=True, thresh=float(p.thresh), kr=float(p.kr) if p.kr > 0 else None,
                skip=skip, interp=skip is not None, biascor=bool(p.biascor),
            )
        vx, vy, vm = vx.T, vy.T, vm.T
        bad = vm == 0
        self.last = {"invalid_frac": float(bad.mean())}
        flow = np.dstack([vx, vy]).astype(np.float32)
        flow[bad] = 0
        return np.nan_to_num(flow, copy=False)


ESTIMATORS = {"tvl1": TVL1Estimator, "piv": PIVEstimator, "flct": FLCTEstimator}


def make_estimator(params):
    """Estimator for a ``TVL1Params``, ``PIVParams`` or ``FLCTParams``."""
    return ESTIMATORS[params.METHOD](params)
