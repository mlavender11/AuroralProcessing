
def add_parser(sub):
    p = sub.add_parser(
        "video",
        help="Make mp4 video from HiST HDF5 file",
        description="Make a video from a HiST HDF5 file at the source frame rate (no binning or averaging). "
        "Used for looking at raw data and pre-processing outputs.",
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
