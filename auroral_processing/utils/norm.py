import numpy as np
import h5py
from tqdm.auto import tqdm
from matplotlib.colors import LogNorm, Normalize
from datetime import datetime
from pathlib import Path
from auroral_processing.utils.timing import get_start_end_idx, _assert_utc


def compute_norm(
    imgs,
    ut_time,
    sample_interval_seconds,
    start_idx=0,
    end_idx=None,
    *,
    low_percentile=1,
    high_percentile=99,
    chunk_size=50,
    linear=False,  # add to docstring
):
    """
    Generates a matplotlib LogNorm based on inputted array of frames
    Samples at even intervals based on sample_interval_seconds

    Parameters
    ----------
    imgs : np.ndarray or h5py.Dataset
        Array of frames, shape (n_frames, height, width). Each pixel must be a 16-bit unsigned integer.
    ut_time : np.ndarray or h5py.Dataset
        Unix epoch time, corresponding to each frame in imgs.
    sample_interval_seconds : float
        How often, in seconds, to sample a frame.
    start_idx : int, optional
        Start index for sample, by default 0.
    end_idx : _type_, optional
        End index for sample, defaults to the last index of imgs/ut_time
    low_percentile : int, optional
        vmin percentile value for LogNorm, by default 1
    high_percentile : int, optional
        vmax percentile value for LogNorm, by default 99
    chunk_size : int, optional
        Splits calculations into chunks to save memory, by default 50

    Returns
    -------
    matplotlib.colors.LogNorm
        normalization function

    Raises
    ------
    ValueError
        If data isn't stored as 16-bit unsigned integers
    ValueError
        If no nonzero pixels are found in the sampled frames
    """

    if end_idx is None:
        end_idx = imgs.shape[0] - 1

    duration_seconds = ut_time[end_idx] - ut_time[start_idx]
    n_samples = int(duration_seconds / sample_interval_seconds)
    sample_idx = np.linspace(start_idx, end_idx, n_samples, dtype=int)

    if imgs.dtype != np.uint16:
        raise ValueError(f"Expected uint16 image data, got {imgs.dtype}")

    hist = np.zeros(65536, dtype=np.int64)

    chunk_starts = range(0, len(sample_idx), chunk_size)
    for chunk_start in tqdm(chunk_starts, desc="computing norm", unit="chunk"):
        chunk_idx = sample_idx[chunk_start : chunk_start + chunk_size]
        chunk = imgs[chunk_idx]
        vals = chunk.ravel()
        vals = vals[vals > 0]
        hist += np.bincount(vals, minlength=65536)

    cdf = np.cumsum(hist)
    total = cdf[-1]

    if total == 0:
        raise ValueError("No nonzero pixels found in sampled frames")

    def value_at_percentile(p):
        target = total * p / 100
        return np.searchsorted(cdf, target)

    if linear:
        vmin = value_at_percentile(low_percentile)
        vmax = value_at_percentile(high_percentile)
        return Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin = max(value_at_percentile(low_percentile), 1e-6)
        vmax = value_at_percentile(high_percentile)
        return LogNorm(vmin=vmin, vmax=vmax)


def compute_norm_from_hdf(
    hdf_fn,
    sample_interval_seconds,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    *,
    low_percentile=1,
    high_percentile=99,
    chunk_size=50,
    linear=False,
):
    if start_time is not None:
        _assert_utc(start_time)
    if end_time is not None:
        _assert_utc(end_time)

    hdf_path = Path(hdf_fn)

    with h5py.File(hdf_path, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"][:]

        if start_time is None:
            start_time = datetime.fromtimestamp(ut[0], datetime.timezone.utc)
        if end_time is None:
            end_time = datetime.fromtimestamp(ut[-1], datetime.timezone.utc)

        start_idx, end_idx = get_start_end_idx(start_time, end_time, ut)

        norm = compute_norm(
            imgs,
            ut,
            sample_interval_seconds,
            start_idx=start_idx,
            end_idx=end_idx,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            chunk_size=chunk_size,
            linear=linear,
        )

    return norm
