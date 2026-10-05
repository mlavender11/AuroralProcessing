# Command line script for making videos from hist HDF5 files

import argparse
from auroral_processing import hdf_utils
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import datetime
import numpy as np
from tqdm.auto import tqdm
import imageio
from auroral_processing.consumers.video import VideoConsumer

parser = argparse.ArgumentParser(description="Make video from HiST HDF5 file")
parser.add_argument("filename", type=str, help="path to HDF5 file")
parser.add_argument("-o", "--out_path", type=str, help="output path of video. Defaults to input with .mp4 suffix")

args = parser.parse_args()

if not args.out_path:
    out_path = Path(args.filename).with_suffix(".mp4")
else:
    out_path = Path(args.out_path).mkdir(parents=True, exist_ok=True)


hdf_path = Path(args.filename)
cmap = plt.get_cmap("gray")
font = hdf_utils.get_font(size=16)

with h5py.File(hdf_path, "r") as f:
    imgs = f["rawimg"]
    ut = f["ut1_unix"][:]
    # imgs = f["pipeline-full"] TODO make this a parameter
    # ut = f["pipeline-full-ut1_unix"]
    n_frames, height, width = imgs.shape

    start_time = datetime.datetime.fromtimestamp(ut[0], datetime.timezone.utc)
    end_time = datetime.datetime.fromtimestamp(ut[-1], datetime.timezone.utc)

    cadence = np.median(np.diff(ut))
    fps = 1 / cadence

    norm = hdf_utils.compute_norm(imgs, ut, 1)
    frame_to_rgb = hdf_utils.get_frame_to_rgb(cmap, norm)

    # with imageio.get_writer(out_path, format="FFMPEG", fps=fps, codec="libx264", quality=6) as writer:
    # with imageio.get_writer(out_path, format="FFMPEG", fps=fps, codec="h264_videotoolbox", bitrate="8M") as writer:
    with imageio.get_writer(out_path, format="FFMPEG", fps=fps, codec="libx264", quality=9) as writer:
        video = VideoConsumer(writer, font, frame_to_rgb, height, width, imgs.dtype, ut.dtype, bin_size=1)

        for n in tqdm(range(n_frames), desc=str(out_path), unit="frame"):
            frame, frame_time = imgs[n], ut[n]
            video.update(n, frame, frame_time)

        video.finalize()
