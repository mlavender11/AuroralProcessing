"""
process_night.py -- the unit of work: everything that happens for ONE night.

The whole function is wrapped so that any failure (a corrupt file, a crashed
converter, a read error off a flaky drive) is caught, logged, and the run
moves on to the next night instead of dying.
"""

import matplotlib

matplotlib.use("Agg")

import traceback
from config import (
    OUTPUT_DIR,
    HDF_DIR,
    KEEP_HDF,
    KEEP_MOTION_THRESHOLD,
    HDF_DATASET,
    HDF_TIME,
    PLAYBACK_SPEED,
    BIN_SIZE,
    CMAP,
)
import db
import imageio
import numpy as np
import matplotlib.pyplot as plt
import h5py
import binary_to_hdf

from .hdf_utils import (
    compute_norm,
    get_font,
    get_frame_to_rgb,
    calculate_fps
)

from consumers import VideoConsumer, KeogramConsumer, StatsConsumer, compute_keogram_bins


def process_hdf(f: h5py.File, video_path, keogram_path):
    cmap = plt.get_cmap(CMAP)
    font = get_font()  # TODO what is this doing

    # get data from HDF file
    imgs = f[HDF_DATASET]
    n_frames, height, width = imgs.shape
    ut = f[HDF_TIME][:]

    # get norm
    print("getting norm")
    norm = compute_norm(imgs, ut, 60)
    print("norm complete")

    fps = calculate_fps(ut, BIN_SIZE, PLAYBACK_SPEED)

    # color mapping function
    frame_to_rgb = get_frame_to_rgb(cmap, norm)

    n_keogram_bins, frames_per_bin_keogram = compute_keogram_bins(n_frames, ut)

    keogram = KeogramConsumer(height, width, n_keogram_bins, frames_per_bin_keogram, cmap, norm, keogram_path, ut)
    stats = StatsConsumer(n_frames)

    with imageio.get_writer(
        video_path, format="FFMPEG", fps=fps, codec="libx264", quality=4
    ) as writer:  # USE QUALITY FROM CONFIGF TODO
        video = VideoConsumer(
            writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=20
        )  # TODO revert bin size back to from config
        consumers = [video, keogram, stats]

        for i in range(n_frames):
            frame, time = imgs[i], ut[i]

            for c in consumers:
                c.update(i, frame, time)

            percent_complete = 100 * i / n_frames
            if percent_complete % 1 == 0:
                print(f"processed {i} / {n_frames} -- {percent_complete:.0f}% complete")

        print("finalizing")
        for c in consumers:
            c.finalize()
    print("done, returning stats")
    return stats.results(n_frames)


def process_night(conn, night_id, drive_id, raw_path, *, make_keo_and_vid=True):
    # Idempotency: if it's already finished, skip immediately. This is what
    # makes a restart cheap -- finished nights cost nothing the second time.
    if db.get_status(conn, night_id) == "done":
        print(f"[skip]  {night_id} already done")
        return "skipped"

    db.mark_in_progress(conn, night_id)
    print(f"[start] {night_id}")

    HDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hdf_path = HDF_DIR / f"{night_id}.h5"
    keogram_path = OUTPUT_DIR / f"{night_id}_keogram.png"
    video_path = OUTPUT_DIR / f"{night_id}_summary.mp4"

    try:
        binary_to_hdf.whole_binary_to_hdf(raw_path, hdf_path)

        if make_keo_and_vid:
            with h5py.File(hdf_path, "r") as f:
                stats = process_hdf(f, video_path, keogram_path)

        # Decide whether to keep this night's HDF, based on the mode in config.
        keep = (KEEP_HDF == "all") or (KEEP_HDF == "interesting" and stats["motion"] >= KEEP_MOTION_THRESHOLD)

        if keep:
            kept_hdf = HDF_DIR / f"{night_id}.h5"
            print(f"[keep]  {night_id} -> {kept_hdf}")
        else:
            if hdf_path.exists():
                hdf_path.unlink()  # delete the big transient HDF
            kept_hdf = None

        db.mark_done(conn, night_id, stats, keogram_path, video_path, hdf_path=kept_hdf)
        print(f"[done]  {night_id}  frames={stats['frame_count']} " f"motion={stats['motion']:.3f}")
        return "done"

    except Exception as e:
        # Remove any half-written HDF from scratch so a later retry starts clean.
        if hdf_path.exists():
            try:
                hdf_path.unlink()
            except OSError:
                pass
        db.mark_failed(conn, night_id, traceback.format_exc())
        print(f"[FAIL]  {night_id}: {e}")
        return "failed"


# def process_hdf(f: h5py.File, video_path, keogram_path):
#     cmap = plt.get_cmap(CMAP)
#     font = get_font()

#     # get data from hdf
#     imgs = f[HDF_DATASET]
#     n_frames, height, width = imgs.shape
#     ut = f[HDF_TIME][:]

#     # norm for video and keogram
#     norm = compute_norm(imgs, ut)

#     # fps and frame processing for video
#     source_seconds_per_frame = np.median(np.diff(ut))
#     source_seconds_per_output_frame = source_seconds_per_frame * BIN_SIZE
#     fps = PLAYBACK_SPEED / source_seconds_per_output_frame

#     # color mapping function
#     frame_to_rgb = get_frame_to_rgb(cmap, norm)

#     # keogram utilities
#     slice_idx_NS = width // 2
#     slice_idx_EW = height // 2
#     ut_hours = (ut % 86400) / 3600.0
#     num_seconds = ut[-1] - ut[0]

#     if BIN_WIDTH_SECONDS is None:
#         n_keogram_bins = n_frames
#         frames_per_keogram_bin = 1
#     else:
#         n_keogram_bins = max(1, int(num_seconds / BIN_WIDTH_SECONDS))  # rough target from time
#         frames_per_keogram_bin = max(1, n_frames // n_keogram_bins)  # integer frames per column
#         n_keogram_bins = math.ceil(n_frames / frames_per_keogram_bin)  # actual columns, recomputed

#     keogram_NS = np.zeros((height, n_keogram_bins), dtype=float)
#     keogram_EW = np.zeros((width, n_keogram_bins), dtype=float)

#     # video and keogram buffers
#     video_buffer_frames = np.empty((BIN_SIZE, height, width), dtype=imgs.dtype)
#     video_buffer_times = np.empty(BIN_SIZE, dtype=ut.dtype)
#     video_buffer_idx = 0

#     keogram_buffer_NS = np.empty((frames_per_keogram_bin, height))
#     keogram_buffer_EW = np.empty((frames_per_keogram_bin, width))
#     keogram_buffer_idx = 0
#     keogram_bin_idx = 0

#     # statistics
#     frame_means = np.empty(n_frames, dtype=np.float64)
#     gmax, total, count_px = 0.0, 0.0, 0.0

#     with imageio.get_writer(video_path, format="FFMPEG", fps=fps, codec="libx264", quality=8) as writer:

#         for i in range(n_frames):
#             frame = imgs[i]
#             time = ut[i]

#             # video
#             video_buffer_frames[video_buffer_idx] = frame
#             video_buffer_times[video_buffer_idx] = time
#             video_buffer_idx += 1
#             if video_buffer_idx == BIN_SIZE:
#                 write_video_frame(writer, video_buffer_frames, video_buffer_times, font, frame_to_rgb)
#                 video_buffer_idx = 0

#             # keogram
#             keogram_buffer_NS[keogram_buffer_idx] = frame[:, slice_idx_NS]  # TODO
#             keogram_buffer_EW[keogram_buffer_idx] = frame[slice_idx_EW, :]
#             keogram_buffer_idx += 1
#             if keogram_buffer_idx == frames_per_keogram_bin:
#                 keogram_NS[:, keogram_bin_idx] = keogram_buffer_NS.mean(
#                     axis=0
#                 )  # TODO does this work? Check if works with no binning
#                 keogram_EW[:, keogram_bin_idx] = keogram_buffer_EW.mean(axis=0)
#                 keogram_bin_idx += 1
#                 keogram_buffer_idx = 0

#             # stats
#             frame_means[i] = frame.mean()
#             total += frame.sum()
#             count_px += frame.size
#             gmax = max(gmax, float(frame.max()))

#         if video_buffer_idx > 0:
#             write_video_frame(
#                 writer,
#                 video_buffer_frames[:video_buffer_idx],
#                 video_buffer_times[:video_buffer_idx],
#                 font,
#                 frame_to_rgb,
#             )

#         if keogram_buffer_idx > 0 and keogram_bin_idx < n_keogram_bins:
#             keogram_NS[:, keogram_bin_idx] = keogram_buffer_NS[:keogram_buffer_idx].mean(axis=0)
#             keogram_EW[:, keogram_bin_idx] = keogram_buffer_EW[:keogram_buffer_idx].mean(axis=0)

#     # TODO save keogram
#     save_keogram_NS_and_EW(
#         outfn=keogram_path,
#         keogram_NS=keogram_NS,
#         keogram_EW=keogram_EW,
#         ut_hours=ut_hours,
#         frame_height=height,
#         frame_width=width,
#         cmap=cmap,
#         norm=norm,
#     )

#     # return stats
#     return {
#         "frame_count": n_frames,
#         "max_brightness": gmax,
#         "mean_brightness": total / count_px,
#         "brightness_std": float(frame_means.std()),
#         "motion": float(np.abs(np.diff(frame_means)).mean()),
#     }
