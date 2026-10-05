import h5py
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from functools import partial
from itertools import islice
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import subprocess
import argparse

from auroral_processing.hdf_utils import compute_norm, compute_norm_from_hdf


# Global tools
def get_vmin_vmax(hdf_fn):
    """
    get vmin and vmax values of hdf file
    """
    imgs = hdf_fn["rawimg"]
    ut = hdf_fn["ut1_unix"]
    norm = compute_norm(imgs, ut, 1)
    vmin, vmax = norm.vmin, norm.vmax
    return vmin, vmax


# ------------ Frame level tools -----------------
def apply_norm_to_frame(frame, vmin, vmax, curve="linear"):
    """
    apply normalization to frame based on vmin and vmax, type based on curve

    Parameters
    ----------
    frame :
        video/image frame
    vmin : _type_
        min value for normalization scaling
    vmax : _type_
        max value for normalization scaling
    curve : str, optional
        type of normalization (linear, log, sqrt), by default "linear"

    Returns
    -------
    np.ndarray
        normalized frame, same dimensions as input

    Raises
    ------
    ValueError
        if curve is not of a valid type
    """
    frame = np.clip(frame, vmin, vmax).astype(np.float32)
    if curve == "linear":
        return (frame - vmin) / (vmax - vmin)
    elif curve == "log":
        return (np.log(frame) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    elif curve == "sqrt":
        return (np.sqrt(frame) - np.sqrt(vmin)) / (np.sqrt(vmax) - np.sqrt(vmin))
    raise ValueError(curve)


def apply_spatial_median_to_frame(frame, k=3):
    """
    use cv2.medianBlur on frame - applies spatial blurring

    Parameters
    ----------
    frame : np.ndarray
        input frame
    k : int, optional
        kernel size, must be odd, by default 3

    Returns
    -------
    np.ndarray
        blurred frame, same size as input
    """
    return cv2.medianBlur(frame, k)


def apply_temporal_median(frame_stack):
    """
    Apply temporal median to a stack of frames

    Parameters
    ----------
    frame_stack : list[np.ndarray]
        stack of frames to merge together

    Returns
    -------
    np.ndarray
        frame the same size as input
    """
    return np.median(frame_stack, axis=2)


# -------------- Generators --------------------
# chained together to apply effects


def read_h5(frame_dataset, block=64):
    """
    read in HDF5 file frame by frame
    reads frams from HDF5 in blocks

    Parameters
    ----------
    frame_dataset :
        dataset within HDF5 file containing frames
    block : int, optional
        number of frames to read from drive at once, by default 64

    Yields
    ------
    individual frames
    """
    for start in range(0, len(frame_dataset), block):
        # pull of size chunk, for memory performance
        chunk = frame_dataset[start : start + block]
        for frame in chunk:
            yield frame


def normalize(src, vmin, vmax, curve="linear"):
    """
    apply normalize to sequence of frames, pulled from an iterable

    Parameters
    ----------
    src : iterable
        iterable containing frames, intended to be np.ndarray
    vmin : float
        vmin value used for normalization
    vmax : float
        vmax value used for normalization
    curve : str, optional
        curve type, see apply_norm_to_frame, by default "linear"

    Yields
    ------
    individual frame
    """
    for frame in src:
        yield apply_norm_to_frame(frame, vmin, vmax, curve=curve)


def spatial_median(src, k=3):
    """
    apply spatial median to a sequence of frames, pulled from an iterable

    Parameters
    ----------
    src : iterable
        iterable containing frames, intended to be np.ndarray
    k : int, optional
        kernel size for median, must be odd, by default 3

    Yields
    ------
    individual frame
    """
    # TODO k > 5 for float32? Maybe use scipy?
    for frame in src:
        yield apply_spatial_median_to_frame(frame, k)


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


def frame_binning_avg(src, bin_size=15, frame_shape=(512, 512)):
    counter = 0
    frame_sum = None
    for frame in src:
        if frame_sum is None:
            frame_sum = np.zeros(frame_shape, dtype=np.float64)

        # Accumulate frames, continue until bin is full
        frame_sum += frame
        counter += 1

        # if bin is full, average together and reset
        if counter == bin_size:
            frame_avg = frame_sum / bin_size
            yield frame_avg
            frame_sum, counter = None, 0


def to_uint8(src, lo=0.0, hi=1.0):
    scale = 255.0 / (hi - lo)
    for frame in src:
        clipped = np.clip((frame - lo) * scale, 0, 255)
        rounded = np.rint(clipped)
        int_frame = rounded.astype(np.uint8)
        yield int_frame


# Pipeline
def chain(src, *stages):
    # src is the original source,
    # the source of each stage is the prior stage
    for stage in stages:
        src = stage(src)
    # return the final stage, pull from it to operate the chain
    return src


# driver - used to pull from the end of a chain of generator stages
def write_h5(src, hdf_file, dataset_name, total_frames, dtype=np.uint8, compression=None, block=64, attrs=None):
    f = hdf_file

    if dataset_name in f:
        del f[dataset_name]  # TODO implement protections
        # raise ValueError("Can't overwrite a dataset")

    dataset = None
    i = 0
    for frame in tqdm(src, total=total_frames):
        if dataset is None:
            frame_shape = frame.shape
            dataset = f.create_dataset(
                dataset_name,
                shape=(total_frames,) + frame_shape,
                dtype=dtype,
                chunks=(1,) + frame_shape,
                compression=compression,
            )
            for key, val in (attrs or {}).items():
                dataset.attrs[key] = val  # what is this?

        dataset[i] = frame
        i += 1

    if dataset is not None and dataset.shape[0] != i:
        dataset.resize(i, axis=0)

    return i


# Running the chain
def apply_processing(hdf_fn):
    with h5py.File(hdf_fn, "r+") as f:
        frames = f["rawimg"]
        vmin, vmax = get_vmin_vmax(f)  # TODO get norm
        total_frames = len(frames)

        bin_size = 15
        pipeline = chain(
            read_h5(frames),
            partial(normalize, vmin=vmin, vmax=vmax),
            partial(spatial_median, k=3),
            partial(frame_binning_avg, bin_size),
            to_uint8,
        )

        n = write_h5(pipeline, f, "pipeline-full", total_frames=total_frames // bin_size, compression="lzf")
        print(f"wrote {n} frames")


def repack(hdf_fn, replace=False):
    src_fn = Path(hdf_fn)
    tmp_fn = src_fn.with_suffix(".repacked.h5")

    with h5py.File(src_fn, "r") as src, h5py.File(tmp_fn, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for name in src:
            print(f"copying {name}")
            src.copy(name, dst)

        old, new = src_fn.stat().st_size, tmp_fn.stat().st_size
        print(f"{old/1e9:.2f} GB -> {new/1e9:.2f} GB (saved {(old-new)/1e6:.0f} MB)")

        if replace:
            tmp_fn.replace(src_fn)
        return tmp_fn


# Video creation
def make_video(*, hdf_fn, out_fn, video_quality=6, playback_speed=None, fps=None, norm=None):
    import imageio
    from auroral_processing.consumers.video import VideoConsumer

    hdf_fn = Path(hdf_fn)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("gray")

    with h5py.File(hdf_fn, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"]
        n_frames, height, width = imgs.shape

        if norm is None:
            norm = compute_norm(hdf_fn, ut, 1)

        # apply norm
        with imageio.get_writer(out_fn, format="FFMPEG", fps=fps, codec="libx264", quality=video_quality) as writer:
            video = VideoConsumer(
                writer,
                None,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-Processing frames in HDF5 file")
    parser.add_argument("hdf_fn", help="path to the HDF5 file")
    parser.add_argument("-r", "--repack", action="store_true", help="repack the HDF file")
    parser.add_argument(
        "-i",
        "--inplace",
        action="store_true",
        help="with --repack: overwrite the original file instead of writing seperate .repacked.h5",
    )
    args = parser.parse_args()

    apply_processing(args.hdf_fn)
    if args.repack:
        print("repacking")
        repack(args.hdf_fn, replace=args.inplace)
