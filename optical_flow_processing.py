import h5py
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from functools import partial
from itertools import islice
from tqdm.auto import tqdm

from hdf_utils import compute_norm, compute_norm_from_hdf


# Global tools
def get_vmin_vmax(hdf_f):
    imgs = hdf_f["rawimg"]
    ut = hdf_f["ut1_unix"]
    norm = compute_norm(imgs, ut, 1)
    vmin, vmax = norm.vmin, norm.vmax
    return vmin, vmax


# Frame level tools
def apply_norm_to_frame(frame, vmin, vmax, curve="linear", beta=None):
    frame = np.clip(frame, vmin, vmax).astype(np.float32)
    if curve == "linear":
        return (frame - vmin) / (vmax - vmin)
    if curve == "log":
        return (np.log(frame) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    if curve == "sqrt":
        return (np.sqrt(frame) - np.sqrt(vmin)) / (np.sqrt(vmax) - np.sqrt(vmin))
    if curve == "asinh":
        b = beta or max(vmin, 1.0)
        return np.arcsinh((frame - vmin) / b) / np.arcsinh((vmax - vmin) / b)
    raise ValueError(curve)


def apply_spatial_median_to_frame(frame, k=3):
    return cv2.medianBlur(frame, k)


def apply_temporal_median(frame_stack):
    return np.median(frame_stack, axis=2)


# Generators
def read_h5(frame_dataset, block=64):
    for start in range(0, len(frame_dataset), block):
        chunk = frame_dataset[start : start + block]
        for frame in chunk:
            yield frame


def normalize(src, vmin, vmax, curve="linear"):
    for frame in src:
        yield apply_norm_to_frame(frame, vmin, vmax)


def spatial_median(src, k=3):
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
        if n > window:
            yield np.partition(ring, k, axis=0)[k]


def frame_binning_avg(src, bin_size=15, frame_shape=(512, 512)):
    counter = 0
    frame_sum = np.zeros(frame_shape)
    for frame in src:
        if counter == bin_size:
            frame_avg = frame_sum / bin_size
            frame_sum = np.zeros_like(frame_sum)
            counter = 0
            yield frame_avg
        else:
            frame_sum += frame
            counter += 1


def to_uint8(src, lo=0.0, hi=1.0):
    scale = 255.0 / (hi - lo)
    for frame in src:
        clipped = np.clip((frame - lo) * scale, 0, 255)
        int_frame = clipped.astype(np.uint8)
        rounded = np.rint(int_frame)
        yield rounded


# Pipeline
def chain(src, *stages):
    for stage in stages:
        src = stage(src)
    return src


# driver
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

        pipeline = chain(
            read_h5(frames),
            partial(normalize, vmin=vmin, vmax=vmax),
            partial(spatial_median, k=3),
            partial(frame_binning_avg),
            to_uint8,
        )

        n = write_h5(pipeline, f, "pipeline-full", total_frames=total_frames, compression="lzf")
        print(f"wrote {n} frames")


if __name__ == "__main__":
    hdf_fn = "/Users/michaellavender/Documents/BUSPC/optical flow/2013-03-27-CamSer7196-opt-flow-subset-copy.h5"
    apply_processing(hdf_fn)


# Video creation
def make_video(*, hdf_fn, out_fn, video_quality=6, playback_speed=None, fps=None, norm=None):
    import imageio
    from video_consumer import VideoConsumer

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
