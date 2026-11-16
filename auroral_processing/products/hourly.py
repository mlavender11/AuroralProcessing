from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from auroral_processing.consumers.keogram_hourly import HourlyKeogramConsumer
from auroral_processing.consumers.video import VideoConsumer
from auroral_processing.io.build_paths import build_output_paths
from auroral_processing.pipeline.frame_ops import get_frame_to_rgb
from auroral_processing.products.video import assert_video_parameters, get_font
from auroral_processing.utils.norm import compute_norm
from auroral_processing.utils.timing import _assert_utc, find_closest_item, get_hourly_boundaries, get_start_end_idx


def make_hourly_videos_keograms(
    *,
    hdf_path,
    out_dir,
    camera=None,
    serial=None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    bin_size=None,
    video_quality=6,
    playback_speed=None,
    fps=None,
    norm=None,
    sample_rate_hz=None,
    font_size=16,
    make_keogram=True,
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
    camera : str, optional
        Camera name used in output file names (e.g. "DMC", "HST"). If None,
        outputs are named after the HDF5 file. See build_output_paths.
    serial : str, optional
        Camera serial, used in output file names when camera is "HST".
    start_time : datetime.datetime, optional
        UTC start of the range to render. Defaults to the first frame's
        timestamp.
    end_time : datetime.datetime, optional
        UTC end of the range to render. Defaults to the last frame's
        timestamp.
    bin_size : int, optional
        Number of source frames averaged into each output video frame. When
        sample_rate_hz is given, must not exceed the downsampling stride.
    video_quality : int, optional
        FFMPEG quality setting passed to imageio, by default 6.
    playback_speed : float, optional
        Playback speed multiplier. See assert_video_parameters.
    fps : float, optional
        Output frames per second. See assert_video_parameters.
    norm : matplotlib.colors.Normalize, optional
        Normalization to apply to pixel values. Computed automatically from a
        sample of the frame range if not given.
    sample_rate_hz : float, optional
        If given, the video is downsampled so only 1 in every N source
        frames is written, where N is chosen to hit this output rate.
        Keograms still include every frame regardless of this setting.
    font_size : int, optional
        Size of the timestamp text on video frames, by default 16.
    make_keogram : bool, optional
        Also write an hourly keogram PNG, by default True.

    Returns
    -------
    matplotlib.colors.Normalize
        The normalization used (either the one passed in or the one
        computed).

    Raises
    ------
    ValueError
        If sample_rate_hz is given and bin_size exceeds the downsampling stride.
    """
    if start_time is not None:
        _assert_utc(start_time)
    if end_time is not None:
        _assert_utc(end_time)

    hdf_path = Path(hdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("gray")
    font = get_font(size=font_size)

    with h5py.File(hdf_path, "r") as f:
        imgs = f["rawimg"]
        ut = f["ut1_unix"][:]
        n_frames, height, width = imgs.shape

        if start_time is None:
            start_time = datetime.fromtimestamp(ut[0], timezone.utc)
        if end_time is None:
            end_time = datetime.fromtimestamp(ut[-1], timezone.utc)

        playback_speed, fps, bin_size = assert_video_parameters(playback_speed, fps, bin_size, ut)

        # downsampling: keep the first bin_size frames of every stride
        if sample_rate_hz is not None:
            native_fps = 1 / float(np.median(np.diff(ut)))
            stride = max(1, round(native_fps / sample_rate_hz))
            fps = sample_rate_hz * playback_speed
            if bin_size > stride:
                raise ValueError(f"bin size {bin_size} must not exceed stride {stride}")
        else:
            stride = None  # no downsampling, every frame goes to the video

        # hour boundaries as datetimes, so each segment's clock hour comes from
        # the boundary itself rather than from the (possibly early) nearest frame
        hours = get_hourly_boundaries(start_time, end_time)
        sub_idx = [find_closest_item(ut, h.timestamp()) for h in hours]

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

        segments = list(pairwise(sub_idx))
        for i, (sub_start_idx, sub_end_idx) in tqdm(list(enumerate(segments)), desc="videos", unit="video"):
            # end-exclusive, except the last segment keeps the final frame
            if i == len(segments) - 1:
                sub_end_idx += 1
            if sub_end_idx <= sub_start_idx:
                continue  # no frames in this segment

            hour_start = hours[i].replace(minute=0, second=0, microsecond=0)
            video_fn, keogram_fn = build_output_paths(
                hdf_path, out_dir, hour_start.timestamp(), i, camera=camera, serial=serial
            )

            with imageio.get_writer(
                video_fn, format="FFMPEG", fps=fps, codec="libx264", quality=video_quality
            ) as writer:
                video = VideoConsumer(writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=bin_size)

                if make_keogram:
                    keogram = HourlyKeogramConsumer(
                        height=height,
                        width=width,
                        cmap=cmap,
                        norm=norm,
                        outfn=keogram_fn,
                        hour_start_ut=hour_start.timestamp(),
                    )

                for n in tqdm(range(sub_start_idx, sub_end_idx), desc=video_fn.stem, unit="frame", leave=False):
                    frame, frame_time = imgs[n], ut[n]

                    if stride is None or (n - sub_start_idx) % stride < bin_size:
                        video.update(n, frame, frame_time)

                    if make_keogram:
                        keogram.update(n, frame, frame_time)

                video.finalize()
                if make_keogram:
                    keogram.finalize()

        return norm
