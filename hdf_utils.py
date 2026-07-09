import matplotlib

matplotlib.use("Agg")
import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from PIL import ImageFont
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


def get_frame_to_rgb(cmap, norm):
    """
    Build a function that converts a raw frame to an RGB uint8 image.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Color map applied to normalized pixel values.
    norm : matplotlib.colors.Normalize
        Normalization applied to raw pixel values before colormapping.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that maps a raw frame array to an (H, W, 3) uint8 RGB array.
    """

    def frame_to_rgb(raw_frame):
        rgba = cmap(norm(raw_frame))
        rgb = rgba[:, :, :3]
        return (rgb * 255).astype(np.uint8)

    return frame_to_rgb


# TODO comment
def get_font(size=16):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()

    return font


def find_closest_item(arr, key):
    """
    Finds the index of the element in arr that is closest in value to key

    Parameters
    ----------
    arr : array_like
        Sorted (ascending) 1D array to be searched
    key : float
        Value to find closest element to.

    Returns
    -------
    int
        Index of element in arr that is closest to key.
    """
    idx = np.searchsorted(arr, key)
    if idx == 0:
        return idx
    if idx == len(arr):
        return idx - 1

    left, right = idx - 1, idx
    return left if abs(arr[left] - key) <= abs(arr[right] - key) else right


def get_start_end_idx(start_time: datetime, end_time: datetime, unix_list):
    """
    Given a start time, end time, and unix epoch time list, returns index of the start and end time

    Parameters
    ----------
    start_time : datetime.datetime
        UTC start time to locate in unix_list
    end_time : datetime.datetime
        UTC end time to locate in unix_list
    unix_list : array_like
        Sorted (ascending) list of unix epoch time values.

    Returns
    -------
    tuple[int, int]
        Index of the frame closest to start_time and index of the frame
        closest to end_time.
    """

    _assert_utc(start_time)
    _assert_utc(end_time)

    start_time_unix = int(start_time.timestamp())
    end_time_unix = int(end_time.timestamp())

    start_idx = find_closest_item(unix_list, start_time_unix)
    end_idx = find_closest_item(unix_list, end_time_unix)

    return start_idx, end_idx


def _assert_utc(dt: datetime.datetime) -> None:
    """
    Asserts that a datetime object is in UTC.

    Parameters
    ----------
    dt : datetime.datetime
        Datetime object to be checked.

    Raises
    ------
    ValueError
        Raises if dt is not timezone-aware and in UTC.
    """
    if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"datetime must be timezone-aware and in UTC, got: {dt!r}")


def calculate_fps(ut, bin_size, playback_speed):
    """
    Calculates the fps of output video based on bin size and playback speed.

    Parameters
    ----------
    ut : array_like
        List of unix epoch time, corresponding to each frame
    bin_size : int
        Number of source frames in each bin, to be averaged together into one output frame.
    playback_speed : float
        Playback speed multiplier.

    Returns
    -------
    float
        Output frames per second.
    """
    source_seconds_per_frame = np.median(np.diff(ut))
    source_seconds_per_output_bin = source_seconds_per_frame * bin_size
    fps = playback_speed / source_seconds_per_output_bin

    return fps


def calculate_bin_size(ut, fps, playback_speed):
    """
    Calculates the frame bin size required for specified output fps and playback speed
    bins are collections of source frames to be averaged into one output frame.

    Parameters
    ----------
    ut : array_like
        List of unix epoch time, corresponding to each frame.
    fps : float
        Output frames per second.
    playback_speed : float
        Playback speed multiplier.

    Returns
    -------
    int
        Frames per bin. Minimum of one.
    """

    source_seconds_per_frame = np.median(np.diff(ut))
    bin_size = playback_speed / (fps * source_seconds_per_frame)
    return max(1, round(bin_size))


def assert_video_parameters(playback_speed, fps, bin_size, ut):
    """
    Asserts that exactly two out of playback speed, fps, and bin size are entered,
    computes the third value, and returns all three.

    Parameters
    ----------
    playback_speed : floar or None
        Playback speed multiplier.
    fps : float or None
        Output frames per second.
    bin_size : int or None
        Number of source frames in each bin, to be averaged together into one output frame.
    ut : array_like
        List of unix epoch time, corresponding to each frame.

    Returns
    -------
    tuple[float, float, int]
        playback_speed, fps, bin_size

    Raises
    ------
    ValueError
        Less than two parameters given.
    ValueError
        Three parameters given.
    """

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


def get_hourly_boundaries(start_time: datetime.datetime, end_time: datetime.datetime):
    """
    Given a start and end time, return a list containing the start time, the top of every hour until the end time, and the end time.

    Parameters
    ----------
    start_time : datetime.datetime
    end_time : datetime.datetime

    Returns
    -------
    list[datetime.datetime]
        List containing start time, end time, and the top of every hour in
        between, in ascending order.
    """

    if start_time.minute > 0 or start_time.second > 0:
        first_whole_hour = start_time.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        hours = [start_time]
    else:
        first_whole_hour = start_time
        hours = []

    current_hour = first_whole_hour

    while current_hour < end_time:
        hours.append(current_hour)
        current_hour += datetime.timedelta(hours=1)

    hours.append(end_time)

    return hours


def get_hourly_sub_idx(ut, start_time: datetime.datetime, end_time: datetime.datetime):
    """
    Given a start time, end time, and list of unix epoch times, return a list
    containing the indices of the start time, the top of every hour until the
    end time, and the end time.

    Parameters
    ----------
    ut : array_like
        List of unix epoch times.
    start_time : datetime.datetime
    end_time : datetime.datetime

    Returns
    -------
    list[int]
        List containing the indices, in the ut list, of the start time, end
        time, and every hour boundary in between.
    """

    _assert_utc(start_time)
    _assert_utc(end_time)

    hours = get_hourly_boundaries(start_time, end_time)

    hour_idxs = [find_closest_item(ut, hour.timestamp()) for hour in hours]

    return hour_idxs


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
    norm=None,
    make_keogram=False,  # TODO make hourly keograms
):
    """
    Render one MP4 video per hour-aligned segment from an HDF5 raw-frame file.

    Parameters
    ----------
    hdf_path : str or pathlib.Path
        Path to the source HDF5 file, containing "rawimg" and "ut1_unix"
        datasets.
    out_dir : str or pathlib.Path
        Directory to write output videos to; created if it doesn't exist.
    start_time : datetime.datetime, optional
        UTC start of the range to render. Defaults to the first frame's
        timestamp.
    end_time : datetime.datetime, optional
        UTC end of the range to render. Defaults to the last frame's
        timestamp.
    bin_size : int, optional
        Number of source frames averaged into each output frame. Exactly two
        of bin_size, fps, and playback_speed must be provided; see
        assert_video_parameters.
    video_quality : int, optional
        FFMPEG quality setting passed to imageio, by default 6.
    playback_speed : float, optional
        Playback speed multiplier. See assert_video_parameters.
    fps : float, optional
        Output frames per second. See assert_video_parameters.
    norm : matplotlib.colors.LogNorm, optional
        Normalization to apply to pixel values. Computed automatically from a
        sample of the frame range if not given.
    make_keogram : bool, optional
        Currently unused. By default False.

    Returns
    -------
    matplotlib.colors.LogNorm
        The normalization used (either the one passed in or the one
        computed).
    """

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

        playback_speed, fps, bin_size = assert_video_parameters(playback_speed, fps, bin_size, ut)

        sub_idx = get_hourly_sub_idx(ut, start_time, end_time)

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
            start_idx, end_idx = get_start_end_idx(start_time, end_time, ut)
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


def build_output_paths(hdf_path: Path, out_dir: Path, unix_time, i: int):
    """
    build output paths like CamSer1387_20130414_07.png

    Parameters
    ----------
    hdf_path : Path
        hdf file path
    out_dir : Path
        output director
    unix_time : _type_
        time at beginning of video/keogram subset

    Returns
    -------
    _type_
        video path (stem.mp4), keogram path (stem.png)
    """

    import re

    cam_str = re.search(r"CamSer\d+", hdf_path.stem).group()

    dt = datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc)
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")

    # fn = out_dir / f"{cam_str}_{date_str}_{hour_str}"
    fn = out_dir / f"{cam_str}_{date_str}_{i}"

    return fn.with_suffix(".mp4"), fn.with_suffix(".png")


def make_hourly_videos_keograms(  # TODO check these docstrings
    *,
    hdf_path,
    out_dir,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    bin_size=None,
    video_quality=6,
    playback_speed=None,
    fps=None,
    norm=None,
    output_hz=None,
):
    """
    Render one MP4 video and one keogram PNG per hour-aligned segment, with
    optional temporal downsampling of the video.

    Parameters
    ----------
    hdf_path : str or pathlib.Path
        Path to the source HDF5 file, containing "rawimg" and "ut1_unix"
        datasets.
    out_dir : str or pathlib.Path
        Directory to write output videos and keograms to; created if it
        doesn't exist.
    start_time : datetime.datetime, optional
        UTC start of the range to render. Defaults to the first frame's
        timestamp.
    end_time : datetime.datetime, optional
        UTC end of the range to render. Defaults to the last frame's
        timestamp.
    bin_size : int, optional
        Number of source frames averaged into each output video frame. Must
        be smaller than the downsampling stride. See assert_video_parameters.
    video_quality : int, optional
        FFMPEG quality setting passed to imageio, by default 6.
    playback_speed : float, optional
        Playback speed multiplier. See assert_video_parameters.
    fps : float, optional
        Output frames per second. See assert_video_parameters.
    norm : matplotlib.colors.LogNorm, optional
        Normalization to apply to pixel values. Computed automatically from a
        sample of the frame range if not given.
    output_hz : float, optional
        If given, the video is downsampled so only 1 in every N source
        frames is written, where N is chosen to hit this output rate.
        Keograms still include every frame regardless of this setting.

    Returns
    -------
    matplotlib.colors.LogNorm
        The normalization used (either the one passed in or the one
        computed).

    Raises
    ------
    ValueError
        If bin_size is not smaller than the computed downsampling stride.
    """

    from itertools import pairwise
    from pathlib import Path
    import imageio
    from .consumers import VideoConsumer, HourlyKeogramConsumer

    # from .consumers import VideoConsumer, KeogramConsumer, HourlyKeogramConsumer

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

        playback_speed, fps, bin_size = assert_video_parameters(playback_speed, fps, bin_size, ut)

        # downsampling calculations

        if output_hz is not None:
            native_dt = float(np.median(np.diff(ut)))
            stride = max(1, round((1 / output_hz) / native_dt))
            fps = output_hz * playback_speed
        else:
            stride = 1

        if bin_size > stride:
            raise ValueError(f"bin size {bin_size} must be smaller than stride {stride}")

        # TODO add bin size int verification - must be int

        sub_idx = get_hourly_sub_idx(ut, start_time, end_time)

        if norm is None:
            start_idx, end_idx = get_start_end_idx(start_time, end_time, ut)
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

        for i, (sub_start_idx, sub_end_idx) in tqdm(list(enumerate(pairwise(sub_idx))), desc="videos", unit="video"):
            video_fn, keogram_fn = build_output_paths(hdf_path, out_dir, ut[sub_start_idx], i)

            with imageio.get_writer(
                video_fn, format="FFMPEG", fps=fps, codec="libx264", quality=video_quality
            ) as writer:
                video = VideoConsumer(
                    writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=bin_size
                )
                n_bins, frames_per_bin = compute_keogram_bins(n_frames=sub_end_idx - sub_start_idx, ut=ut)
                # keogram = KeogramConsumer(
                #     height,
                #     width,
                #     n_bins=n_bins,
                #     frames_per_bin=frames_per_bin,
                #     cmap=cmap,
                #     norm=norm,
                #     outfn=keogram_fn,
                #     ut=ut[sub_start_idx:sub_end_idx],
                # )

                if sub_start_idx + 500 >= len(ut):
                    start_hour_idx = len(ut) - 1
                else:
                    start_hour_idx = sub_start_idx + 500

                hour_start_dt = datetime.datetime.fromtimestamp(ut[start_hour_idx], datetime.timezone.utc).replace(
                    minute=0, second=0, microsecond=0
                )
                hour_start_ut = hour_start_dt.timestamp()

                keogram = HourlyKeogramConsumer(
                    height=height, width=width, cmap=cmap, norm=norm, outfn=keogram_fn, hour_start_ut=hour_start_ut
                )

                frame_range = range(sub_start_idx, sub_end_idx)
                for n in tqdm(frame_range, desc=video_fn.stem, unit="frame", leave=False):

                    frame, frame_time = imgs[n], ut[n]

                    if (n - sub_start_idx) % stride < bin_size:
                        video.update(n, frame, frame_time)

                    keogram.update(n, frame, frame_time)
                video.finalize()
                keogram.finalize()

        return norm


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

        norm = compute_norm(imgs, ut, 60)  # does htis need to take sample_interval into account?
        keogram = KeogramConsumer(height, width, n_bins, frames_per_bin_keogram, cmap, norm, keogram_path, ut)

        for start_idx in tqdm(start_idxs[:-1], desc="bins", unit="bin"):
            for i in range(frames_per_bin_keogram):
                frame_idx = start_idx + i
                frame, time = imgs[frame_idx], ut[frame_idx]
                keogram.update(i, frame, time)
        keogram.finalize()


def compute_keogram_bins(n_frames, ut, bin_width_seconds=None):
    """
    Given the number of frames to be processed, a list of unix epoch times of
    each frame, and the width, in seconds, of each bin, return the number of
    bins and the frames in each bin. If bin_width_seconds is None, assumes 1
    frame per bin.

    Parameters
    ----------
    n_frames : int
        Number of frames to be processed.
    ut : array_like
        List of unix epoch times.
    bin_width_seconds : float, optional
        Number of seconds to be averaged together in each bin, by default None.

    Returns
    -------
    tuple[int, int]
        n_bins, frames_per_bin
    """

    import math

    if bin_width_seconds is None:
        return n_frames, 1
    num_seconds = ut[-1] - ut[0]
    n_bins = max(1, int(num_seconds / bin_width_seconds))
    frames_per_bin = max(1, n_frames // n_bins)
    n_bins = math.ceil(n_frames / frames_per_bin)  # actual columns
    return n_bins, frames_per_bin
