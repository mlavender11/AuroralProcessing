"""
CLI Interface for extract_clip in auroral_processing.io.hdf_utilities
"""

import argparse
from auroral_processing.utils.timing import get_start_end_datetime
from auroral_processing.io.hdf_utilities import extract_clip


def main():
    parser = argparse.ArgumentParser(description="Extract subset of HDF5 video file")
    parser.add_argument("filename", help="HDF5 input file")
    parser.add_argument("start_time", help="time to start output clip, HH:MM:SS")  # TODO
    parser.add_argument("end_time", help="time to end output clip, HH:MM:SS")
    parser.add_argument(
        "-o", "--output_filename", default=None, help="Filename of output. Defaults to input with times appended."
    )
    args = parser.parse_args()

    # start and end times in utc format
    start_time, end_time = get_start_end_datetime(args.filename, args.start_time, args.end_time)
    extract_clip(args.filename, start_time, end_time, args.output_filename)


if __name__ == "__main__":
    main()
