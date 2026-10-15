"""
Processing stages, chained together with processing_chain.chain().

Every stage takes and yields (t, frame) pairs, so each frame carries its own
timestamp through the chain and the time dataset written at the end always
matches the frames:

  - per-frame stages (normalize, spatial_median, apply_wiener, to_uint16)
    pass t through unchanged
  - frame_binning_avg     -> mean time of the frames in the bin (bin centre)
  - sliding_temporal_median -> time of the centre frame of the window

"""

from collections import deque

import numpy as np
from scipy.signal import wiener

from auroral_processing.pipeline.frame_ops import apply_norm_to_frame, apply_spatial_median


def read_h5(frame_dataset, time_dataset=None, block=64):
    """
    Read an HDF5 image dataset frame by frame, paired with its timestamps.

    Parameters
    ----------
    frame_dataset : h5py.Dataset
        frames, shape (n_frames, ny, nx)
    time_dataset : h5py.Dataset or array-like, optional
        one timestamp per frame (e.g. ut1_unix). If None, t is the frame index.
    block : int, optional
        number of frames to read from disk at once, by default 64

    Yields
    ------
    (t, frame) : (float, np.ndarray)
    """
    n = len(frame_dataset)
    if time_dataset is not None and len(time_dataset) != n:
        raise ValueError(f"time dataset {time_dataset.name} has {len(time_dataset)} entries, frame dataset {frame_dataset.name} has {n}")

    for start in range(0, n, block):
        # pull a block at a time, for memory performance
        chunk = frame_dataset[start : start + block]
        if time_dataset is not None:
            tchunk = np.asarray(time_dataset[start : start + block], dtype=np.float64)
        else:
            tchunk = np.arange(start, start + len(chunk), dtype=np.float64)
        for t, frame in zip(tchunk, chunk):
            yield t, frame




def normalize(src, vmin, vmax, curve="linear"):
    """
    Normalize each frame, see apply_norm_to_frame.

    Parameters
    ----------
    src : iterable of (t, frame)
    vmin, vmax : float
        normalization limits
    curve : str, optional
        curve type, see apply_norm_to_frame, by default "linear"
    """
    for t, frame in src:
        yield t, apply_norm_to_frame(frame, vmin, vmax, curve=curve)


def spatial_median(src, k=3):
    """
    Spatial median filter on each frame.

    Parameters
    ----------
    src : iterable of (t, frame)
    k : int, optional
        kernel size, must be odd, by default 3
    """
    # TODO k > 5 for float32? Maybe use scipy?
    for t, frame in src:
        yield t, apply_spatial_median(frame, k)


def sliding_temporal_median(src, window=5):  # TODO need to pad edges?
    # Must window be odd?
    if window % 2 == 0:
        raise ValueError(f"window must be odd: current window size = {window}")

    k = window // 2
    ring, n = None, 0
    for frame in src:
        if ring is None:  # Why not do this above?
            ring = np.empty((window,) + frame.shape, dtype=np.float32)

        ring[n % window] = frame  # why this and not overwrite n?
        n += 1
        if n >= window:
            yield np.partition(ring, k, axis=0)[k]


def frame_binning_avg(src, bin_size=15, keep_partial=False, frame_shape=None):
    """
    Average consecutive non-overlapping bins of `bin_size` frames.

    The output time is the mean of the bin's input times (the bin centre for
    a regular cadence), so frame-to-frame dt of the output is bin_size times
    the input cadence.

    Parameters
    ----------
    src : iterable of (t, frame)
    bin_size : int
        frames per bin
    keep_partial : bool, optional
        also yield a final, shorter bin if frames are left over; by default
        they are dropped (output has n // bin_size frames)
    frame_shape : ignored
        kept for backwards compatibility; the shape is taken from the frames
    """
    frame_sum = None
    t_ref, dt_sum, counter = 0.0, 0.0, 0
    for t, frame in src:
        if frame_sum is None:
            frame_sum = np.zeros(frame.shape, dtype=np.float64)
            t_ref = t  # sum offsets from the bin's first time, to keep precision

        # accumulate frames until the bin is full
        frame_sum += frame
        dt_sum += t - t_ref
        counter += 1

        # bin full: average together and reset
        if counter == bin_size:
            yield t_ref + (dt_sum / bin_size), frame_sum / bin_size
            frame_sum, dt_sum, counter = None, 0.0, 0

    if keep_partial and counter:
        yield t_ref + (dt_sum / counter), frame_sum / counter


def to_uint16(src, lo=0.0, hi=1.0, out_max=255):
    """
    Map [lo, hi] -> [0, out_max], round and store as uint16.

    out_max defaults to 255 to match the previous behaviour; use 65535 for the
    full uint16 range.
    """
    scale = out_max / (hi - lo)
    for t, frame in src:
        clipped = np.clip((frame - lo) * scale, 0, out_max)
        yield t, np.rint(clipped).astype(np.uint16)


def apply_wiener(src, size=None, noise=None):
    """Wiener filter on each frame, see scipy.signal.wiener."""
    for t, frame in src:
        yield t, wiener(frame, size, noise)
