"""Per-pair quality metrics for a flow field. Images are float32 in 0-1 (see compute_flow.FrameScaler)."""

import numpy as np
import cv2

# Metrics computed by compute_flow. "zero_mae" is the warp residual for zero flow (|I1 - I0|), the
# baseline warp_mae should beat. "fb_error" needs the backward flow, doubling the cost.
METRICS = ("warp_mae", "zero_mae", "fb_error")
DEFAULT_METRICS = ("warp_mae", "zero_mae")


def _sample_coords(flow):
    """Pixel coordinates each flow vector points to, and a mask of those inside the frame."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + flow[..., 0], gy + flow[..., 1]
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return mx, my, valid


def warp_mae(i0, i1, flow):
    """
    Mean absolute brightness difference between I0 and I1 warped back by the flow.

    Only pixels whose flow lands inside the frame are counted. Lower is better; compare with
    ``zero_mae`` for the no-motion baseline.
    """
    mx, my, valid = _sample_coords(flow)
    warped = cv2.remap(i1, mx, my, cv2.INTER_LINEAR)
    return float(np.abs(warped - i0)[valid].mean()) if valid.any() else np.nan


def zero_mae(i0, i1):
    """Mean absolute difference |I1 - I0|: the warp residual of zero flow."""
    return float(np.abs(i1 - i0).mean())


def fb_error(flow, back_flow):
    """
    Forward-backward consistency: mean |f(x) + b(x + f(x))| in px, where b is the I1 -> I0 flow.

    Near 0 where the forward and backward flows agree; large at occlusions and unreliable regions.
    """
    mx, my, valid = _sample_coords(flow)
    b = cv2.remap(back_flow, mx, my, cv2.INTER_LINEAR)
    err = np.hypot(*(flow + b).transpose(2, 0, 1))
    return float(err[valid].mean()) if valid.any() else np.nan
