import argparse

EPILOG = """\
output (-o):
  (none)              next to the input: clip.h5 -> clip.mp4
  name.mp4            a bare name goes next to the input (so does ./name.mp4)
  dir/name.mp4        relative to the current directory, not the input
  /abs/dir/name.mp4   used as given
  Missing folders are created. An existing video is overwritten without asking.
  To write into the current directory, use -o "$PWD/name.mp4".

The video is greyscale, every frame at the frame rate of --times, with the UTC
time drawn on each frame.

Pipeline outputs: `aurora pipeline run ... --preview` already writes a
preview.mp4 into each output folder with the linear scale. Use this command for
a video elsewhere, under another name, or of a file the pipeline didn't make.

examples:
  # raw file, default log scale; writes clip.mp4 next to clip.h5
  aurora video data/clip.h5

  # raw file into a videos folder (created if needed)
  aurora video data/clip.h5 -o videos/clip_raw.mp4

  # pipeline output, shown as stored
  aurora video OUT/frames.h5 --frames frames --times frames_ut1_unix --scale linear -o videos/clip_norm.mp4
"""


def add_parser(sub):
    p = sub.add_parser(
        "video",
        help="Make mp4 video from HiST HDF5 file",
        description="Make a video from a HiST HDF5 file at the source frame rate (no binning or averaging).\n"
        "Used for looking at raw data and pre-processing outputs.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("filename", help="input HDF5 file")
    p.add_argument("-o", "--out_path", default=None,
                   help="output video; a bare name goes next to the input, a path with a folder is used as given "
                        "(default: input with .mp4 suffix)")
    p.add_argument("--frames", default="rawimg", help="frame dataset, default rawimg (pipeline outputs: frames)")
    p.add_argument("--times", default="ut1_unix", help="time dataset, default ut1_unix (pipeline outputs: frames_ut1_unix)")
    p.add_argument("--scale", choices=["log", "linear"], default="log",
                   help="log: log scale between the 1st and 99th percentiles (default, for raw counts); "
                        "linear: 0 to the dtype's maximum, showing frames as stored (for pipeline outputs)")
    p.set_defaults(func=run)


def linear_norm(dtype):
    """Linear 0..max of an integer dtype (0..1 for floats)."""
    import numpy as np
    from matplotlib.colors import Normalize

    vmax = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    return Normalize(vmin=0, vmax=vmax)


def run(args):
    import h5py
    from auroral_processing.products.video import make_video_from_source

    norm = None
    if args.scale == "linear":
        with h5py.File(args.filename, "r") as f:
            norm = linear_norm(f[args.frames].dtype)
    print(make_video_from_source(args.filename, args.out_path, img_dset=args.frames, ut_dset=args.times, norm=norm))
