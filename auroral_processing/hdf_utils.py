def make_hourly_videos_keograms(  # TODO check these docstrings
    *,
    hdf_path,
    out_dir,
    camera,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    bin_size=None,
    video_quality=6,
    playback_speed=None,
    fps=None,
    norm=None,
    sample_rate_hz=None,
    font_size=16,
    make_keogram=True,
    serial=None,
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
    sample_rate_hz : float, optional
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
    from auroral_processing.consumers.video import VideoConsumer
    from auroral_processing.consumers.keogram_hourly import HourlyKeogramConsumer

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
            start_time = datetime.datetime.fromtimestamp(ut[0], datetime.timezone.utc)
        if end_time is None:
            end_time = datetime.datetime.fromtimestamp(ut[-1], datetime.timezone.utc)

        playback_speed, fps, bin_size = assert_video_parameters(playback_speed, fps, bin_size, ut)

        # downsampling calculations
        if sample_rate_hz is not None:
            native_dt = float(np.median(np.diff(ut)))
            native_fps = 1 / native_dt
            stride = max(1, round(native_fps / sample_rate_hz))
            fps = sample_rate_hz * playback_speed
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
            video_fn, keogram_fn = build_output_paths(
                hdf_path, out_dir, ut[sub_start_idx + 30], i, camera=camera, serial=serial
            )

            with imageio.get_writer(
                video_fn, format="FFMPEG", fps=fps, codec="libx264", quality=video_quality
            ) as writer:
                video = VideoConsumer(
                    writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=bin_size
                )
                # n_bins, frames_per_bin = compute_keogram_bins(n_frames=sub_end_idx - sub_start_idx, ut=ut)
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

                if make_keogram:
                    keogram = HourlyKeogramConsumer(
                        height=height, width=width, cmap=cmap, norm=norm, outfn=keogram_fn, hour_start_ut=hour_start_ut
                    )

                frame_range = range(sub_start_idx, sub_end_idx)
                for n in tqdm(frame_range, desc=video_fn.stem, unit="frame", leave=False):

                    frame, frame_time = imgs[n], ut[n]

                    if (n - sub_start_idx) % stride < bin_size:  # Selects the first bin_size frames in every stride
                        video.update(n, frame, frame_time)

                    if make_keogram:
                        keogram.update(n, frame, frame_time)
                video.finalize()
                if make_keogram:
                    keogram.finalize()

        return norm


