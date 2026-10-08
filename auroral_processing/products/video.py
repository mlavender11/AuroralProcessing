import numpy as np
from PIL import ImageFont
from auroral_processing.utils.binning import calculate_bin_size, calculate_fps
from tqdm.auto import tqdm


def get_font(size=16):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()

    return font


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

    return playback_speed, fps, bin_size


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
    from auroral_processing.consumers.video import VideoConsumer

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
