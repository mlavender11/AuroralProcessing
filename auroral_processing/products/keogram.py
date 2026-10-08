from pathlib import Path
import h5py
from auroral_processing.utils.norm import compute_norm
from auroral_processing.utils.binning import compute_keogram_bins
from auroral_processing.consumers.keogram import KeogramConsumer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np


def make_keogram_6_22_26(*, hdf_path, out_dir, bin_width_seconds=None, norm=None):
    """
    Render a single keogram PNG for an entire HDF5 raw-frame file.

    Parameters
    ----------
    hdf_path : str or pathlib.Path
        Path to the source HDF5 file, containing "rawimg" and "ut1_unix"
        datasets.
    out_dir : str or pathlib.Path
        Directory to write the keogram to; created if it doesn't exist. The
        output filename matches hdf_path's stem with a .png suffix.
    bin_width_seconds : float, optional
        Width, in seconds, of each keogram column. Defaults to 1 frame per
        bin.
    norm : matplotlib.colors.LogNorm, optional
        Normalization to apply to pixel values. Computed automatically over
        the whole file if not given.
    """

    cmap = plt.get_cmap("gray")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keogram_path = out_dir / Path(hdf_path).with_suffix(".png").name

    with h5py.File(hdf_path, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"][()]
        n_frames, height, width = imgs.shape

        if norm is None:
            print("getting norm")
            norm = compute_norm(imgs, ut, 1)
            print("norm complete")

        n_keogram_bins, frames_per_bin_keogram = compute_keogram_bins(n_frames, ut, bin_width_seconds)
        keogram = KeogramConsumer(height, width, n_keogram_bins, frames_per_bin_keogram, cmap, norm, keogram_path, ut)

        for i in tqdm(range(n_frames), desc="keogram:", unit="frame"):
            frame, time = imgs[i], ut[i]
            keogram.update(i, frame, time)
        keogram.finalize()


def make_keogram_rougher(*, hdf_path, out_dir, bin_size_seconds, sample_interval):
    """
    Render a keogram PNG using a coarser, faster sampling strategy that only
    reads a subset of frames instead of the full file.

    Parameters
    ----------
    hdf_path : str or pathlib.Path
        Path to the source HDF5 file, containing "rawimg" and "ut1_unix"
        datasets.
    out_dir : str or pathlib.Path
        Directory to write the keogram to; created if it doesn't exist. The
        output filename matches hdf_path's stem with a .png suffix.
    bin_size_seconds : float
        Width, in seconds, of each keogram column.
    sample_interval : int
        Number of source frames to skip between each sampled block. Must be
        larger than the number of frames per keogram bin.

    Raises
    ------
    ValueError
        If frames_per_bin computed from bin_size_seconds is not smaller than
        sample_interval.
    """

    cmap = plt.get_cmap("gray")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keogram_path = out_dir / Path(hdf_path).with_suffix(".png").name

    with h5py.File(hdf_path, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"][()]
        n_frames, height, width = imgs.shape

        start_idxs = np.arange(0, n_frames, sample_interval, dtype=int)

        _, frames_per_bin_keogram = compute_keogram_bins(n_frames, ut, bin_size_seconds)

        if frames_per_bin_keogram >= sample_interval:
            raise ValueError(
                f"frames per bin {frames_per_bin_keogram} must be less than sample interval {sample_interval}"
            )

        n_bins = len(start_idxs) - 1

        norm = compute_norm(imgs, ut, 60)  # does htis need to take sample_interval into account?
        keogram = KeogramConsumer(height, width, n_bins, frames_per_bin_keogram, cmap, norm, keogram_path, ut)

        for start_idx in tqdm(start_idxs[:-1], desc="bins", unit="bin"):
            for i in range(frames_per_bin_keogram):
                frame_idx = start_idx + i
                frame, time = imgs[frame_idx], ut[frame_idx]
                keogram.update(i, frame, time)
        keogram.finalize()
