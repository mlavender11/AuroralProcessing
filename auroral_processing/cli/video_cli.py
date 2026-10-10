import argparse

# from .cli_utils import output_options


# parents=[output_options()],
def add_parser(sub):
    p = sub.add_parser(
        "video",
        help="Make mp4 video from HiST HDF5 file",
        description="Make a video from a HiST HDF File Output is at source frame-rate, no binning or averaging Used for testing flow preprocessing",
    )

    p.add_argument("filename", help="input HiST HDF5 file")
    p.add_argument("-o", "--out_path", default=None, help="output path of video. Defaults to input with .mp4 suffix")
    p.set_defaults(func=run)


def run(args):
    from auroral_processing.products.video import make_video_from_source

    # make_video_from_source(args.filename, args.out_path)
    make_video_from_source(args.filename, args.out_path)
