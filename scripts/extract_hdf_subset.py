"""
Extract subset of HiST HDF5 file based on start and end time
HDF5 file expected to have frames in 'rawimg' and ut time in 'ut1_unix'
"""

import h5py
from pathlib import Path
import argparse
from auroral_processing.hdf_utils import get_start_end_idx
from auroral_processing.HiST import datetime_utils, io, database


def main():
    parser = argparse.ArgumentParser(description="Extract subset of HDF5 video file")
    parser.add_argument("filename", help="HDF5 input file")
    parser.add_argument("start_time", help="time to start output clip in format ...")  # TODO
    parser.add_argument("end_time", help="time to end output clip in format: ...")
    parser.add_argument(
        "-o", "--output_filename", default=None, help="Filename of output. Defaults to input with times appended."
    )
    args = parser.parse_args()

    # start and end times in utc format
    start_time, end_time = datetime_utils.get_start_end_time(args.filename, args.start_time, args.end_time)

    # file naming
    if args.output_filename is None:
        src_fn = Path(args.filename)
        duration_str = f"{start_time:%H%M%S}-{end_time:%H%M%S}"
        out_fn = src_fn.with_name(f"{src_fn.stem}_{duration_str}.h5")
    else:
        out_fn = args.output_filename

    # copy files
    with h5py.File(args.filename, "r") as src_f, h5py.File(out_fn, "w") as out_f:
        start_idx, end_idx = get_start_end_idx(start_time=start_time, end_time=end_time, unix_list=src_f["ut1_unix"])

        io.copy_datasets(src_f, out_f, start_idx=start_idx, end_idx=end_idx)


if __name__ == "__main__":
    main()
