import argparse
from .cli_utils import hhmmss #, output_options

# parents=[output_options()],
def add_parser(sub: argparse.ArgumentParser):
    p = sub.add_parser(
        "extract",
        
        help="extract a time window",
        description="Copy frames between START and END (UTC) into a new HDF file",
    )  # TODO is iti nto a new file or in place?

    p.add_argument("filename", help="input HiST HDF5 file")
    p.add_argument("start", type=hhmmss, help="start time, HH:MM:SS UTC")
    p.add_argument("end", type=hhmmss, help="end time, HH:MM:SS UTC")
    # TODO any others?
    p.set_defaults(func=run)


def run(args):
    from auroral_processing.utils.timing import get_start_end_datetime
    from auroral_processing.io.hdf_utilities import extract_clip

    start_time, end_time = get_start_end_datetime(args.filename, args.start_time, args.end_time)
    extract_clip(args.filename, start_time, end_time, args.output_filename)
