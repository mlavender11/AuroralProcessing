import matplotlib

matplotlib.use("Agg")
import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

_UTC = datetime.UTC


# def compute_norm(
#     imgs, ut_time, sample_interval_seconds, start_idx=0, end_idx=None, *, low_percentile=1, high_percentile=99
# ):
#     if end_idx is None:
#         end_idx = imgs.shape[0] - 1
#     duration_seconds = ut_time[end_idx] - ut_time[start_idx]
#     print("norm udration seconds", duration_seconds)
#     n_samples = int(duration_seconds / sample_interval_seconds)
#     sample_idx = np.linspace(start_idx, end_idx, n_samples, dtype=int)
#     print("sample index done")
#     print("getting sample")
#     sample = imgs[sample_idx].ravel()
#     print("sample done")
#     sample = sample[sample > 0]
#     return LogNorm(vmin=max(np.percentile(sample, low_percentile), 1e-6), vmax=np.percentile(sample, high_percentile))


# With histogram to save memory usage
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
):
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

    vmin = max(value_at_percentile(low_percentile), 1e-6)
    vmax = value_at_percentile(high_percentile)

    return LogNorm(vmin=vmin, vmax=vmax)


def get_frame_to_rgb(cmap, norm):
    def frame_to_rgb(raw_frame):
        rgba = cmap(norm(raw_frame))
        rgb = rgba[:, :, :3]
        return (rgb * 255).astype(np.uint8)

    return frame_to_rgb


def get_font(size=16):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()

    return font


def find_closest_item(arr, key):
    idx = np.searchsorted(arr, key)
    if idx == 0:
        return idx
    if idx == len(arr):
        return idx

    left, right = idx - 1, idx
    return left if abs(arr[left] - key) <= abs(arr[right] - key) else right


def get_start_end_idx(start_time: datetime, end_time: datetime, unix_list):
    _assert_utc(start_time)
    _assert_utc(end_time)

    start_time_unix = int(start_time.timestamp())
    end_time_unix = int(end_time.timestamp())

    start_idx = find_closest_item(unix_list, start_time_unix)
    end_idx = find_closest_item(unix_list, end_time_unix)

    return start_idx, end_idx


def _assert_utc(dt: datetime.datetime) -> None:
    if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"datetime must be timezone-aware and in UTC, got: {dt!r}")


def calculate_fps(ut, bin_size, playback_speed):
    source_seconds_per_frame = np.median(np.diff(ut))
    source_seconds_per_output_frame = source_seconds_per_frame * bin_size
    fps = playback_speed / source_seconds_per_output_frame

    return fps


def calculate_bin_size(ut, fps, playback_speed):
    source_seconds_per_frame = np.median(np.diff(ut))
    bin_size = playback_speed / (fps * source_seconds_per_frame)
    return max(1, round(bin_size))


def assert_video_parameters(playback_speed, fps, bin_size, ut):
    total_given = sum(x is not None for x in [playback_speed, fps, bin_size])
    if total_given < 2:
        raise ValueError(
            f"Only {total_given} parameters entered. Need to input two options: fps = {fps}, playback_speed = {playback_speed}, bin_size = {bin_size}"
        )
    elif total_given == 3:
        raise ValueError(f"Can not enter three parameters. Need to choose two from playback_speed, fps, and bin_size")
    else:
        if playback_speed is None:
            source_seconds_per_frame = np.median(np.diff(ut))
            playback_speed = fps * source_seconds_per_frame * bin_size
        if fps is None:
            fps = calculate_fps(ut, bin_size, playback_speed)
        if bin_size is None:
            bin_size = calculate_bin_size(ut, fps, playback_speed)
            bin_size = max(1, round(bin_size))

    return playback_speed, fps, bin_size


# TODO add guards to pass 2 of three of bin size, playback speed, and fps
def make_video_from_times(
    *,
    hdf_path,
    out_dir,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    bin_size=None,
    video_quality=6,
    playback_speed=None,
    fps=None,
    video_lengths_seconds=None,
    norm=None,
):
    import math
    from itertools import pairwise
    from pathlib import Path

    import imageio

    from .consumers import VideoConsumer

    if start_time is not None:
        _assert_utc(start_time)
    if end_time is not None:
        _assert_utc(end_time)

    hdf_path = Path(hdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("gray")
    font = get_font()

    with h5py.File(hdf_path, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"][:]
        n_frames, height, width = imgs.shape

        if start_time is None:
            start_time = datetime.datetime.fromtimestamp(ut[0], datetime.timezone.utc)
        if end_time is None:
            end_time = datetime.datetime.fromtimestamp(ut[-1], datetime.timezone.utc)

        real_duration_seconds = (end_time - start_time).total_seconds()
        n_videos = math.ceil(real_duration_seconds / video_lengths_seconds) if video_lengths_seconds is not None else 1

        playback_speed, fps, bin_size = assert_video_parameters(playback_speed, fps, bin_size, ut)

        start_idx, end_idx = get_start_end_idx(start_time, end_time, ut)
        # get start/end indicies of each video segment
        sub_idx = np.round(np.linspace(start_idx, end_idx, n_videos + 1)).astype(int)

        n_frames_to_process = end_idx - start_idx

        video_fns = [  # Produce file names for each video. ex. 2013-03-30_8-30-00_9-30-00.mp4
            (
                out_dir
                / (
                    hdf_path.stem
                    + f'_{datetime.datetime.fromtimestamp(ut[s], datetime.timezone.utc).strftime("%H-%M-%S")}_{datetime.datetime.fromtimestamp(ut[e], datetime.timezone.utc).strftime("%H-%M-%S")}'
                )
            ).with_suffix(".mp4")
            for s, e in pairwise(sub_idx)
        ]

        # try getting one norm for entire video
        if norm is None:
            norm = compute_norm(
                imgs,
                ut,
                1,
                start_idx=start_idx,
                end_idx=end_idx,
                low_percentile=0.1,
                high_percentile=99.9,
            )  # TODO try diffferent perecentiles
        frame_to_rgb = get_frame_to_rgb(cmap, norm)

        for (sub_start_idx, sub_end_idx), fn in tqdm(
            list(zip(pairwise(sub_idx), video_fns)), desc="videos", unit="video"
        ):
            with imageio.get_writer(fn, format="FFMPEG", fps=fps, codec="libx264", quality=video_quality) as writer:
                video = VideoConsumer(
                    writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=bin_size
                )
                frame_range = range(sub_start_idx, sub_end_idx + 1)
                for n in tqdm(frame_range, desc=fn.name, unit="frame", leave=False):
                    frame, frame_time = imgs[n], ut[n]
                    video.update(n, frame, frame_time)
                video.finalize()

        return norm


def make_keogram_6_22_26(*, hdf_path, out_dir, bin_width_seconds=None, norm=None):
    from .consumers import KeogramConsumer

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
    from .consumers import KeogramConsumer

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

        print("getting norm")
        norm = compute_norm(imgs, ut, 60)  # does htis need to take sample_interval into account?
        print("norm complete")
        keogram = KeogramConsumer(height, width, n_bins, frames_per_bin_keogram, cmap, norm, keogram_path, ut)

        for start_idx in start_idxs[:-1]:
            for i in range(frames_per_bin_keogram):
                frame_idx = start_idx + i
                frame, time = imgs[frame_idx], ut[frame_idx]
                keogram.update(i, frame, time)

                percent_complete = 100 * frame_idx / n_frames
                if percent_complete % 1 == 0:
                    print(f"processed {frame_idx} / {n_frames} -- {percent_complete:.0f}% complete")

        keogram.finalize()


def utc_time_from_string(): ...  # TODO implement - why do I need this?


def compute_keogram_bins(n_frames, ut, bin_width_seconds):
    import math

    if bin_width_seconds is None:
        return n_frames, 1
    num_seconds = ut[-1] - ut[0]
    n_bins = max(1, int(num_seconds / bin_width_seconds))
    frames_per_bin = max(1, n_frames // n_bins)
    n_bins = math.ceil(n_frames / frames_per_bin)  # actual columns
    return n_bins, frames_per_bin
