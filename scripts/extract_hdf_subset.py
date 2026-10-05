"""
Extract subset of HiST HDF5 file based on start and end time
HDF5 file expected to have frames in 'rawimg' and ut time in 'ut1_unix'
"""

import h5py
import re
import datetime
from pathlib import Path
import argparse
from auroral_processing.hdf_utils import get_start_end_idx
from tqdm.auto import tqdm


def get_date_str(filename):
    # Get the date as a string
    # Expects file name like 2013-03-27-CamSer7196.h5 - just needs to start with YYYY-MM-DD

    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)

    if not match:
        raise ValueError(f"Filename {filename} does not contain a valid date")
    else:
        return match.group(0)


def get_start_end_time(filename, start_time, end_time):
    # Start and end datetime objects in UTC
    # Expects file name like 2013-03-27-CamSer7196.h5 - just needs to start with YYYY-MM-DD
    # start_time and end_time in HH:MM:SS

    # DATE AND START TIME
    tz = datetime.timezone.utc
    date_str = get_date_str(Path(filename).name)
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    # Start and end time from strings
    start_time = datetime.datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.datetime.strptime(end_time, "%H:%M:%S").time()

    # Timezone aware start and end timestamps in UTC format (with date)
    start = datetime.datetime.combine(date, start_time, tzinfo=tz)
    end = datetime.datetime.combine(date, end_time, tzinfo=tz)

    return start, end


def copy_datasets(src: h5py.File, dst: h5py.File, start_idx, end_idx):
    # Copys HiST HDF5 datasets from start_idx to end_idx
    # expects opened HDF5 files

    if end_idx < start_idx:
        raise ValueError(f"Empty frame range: start_idx={start_idx}, end_idx={end_idx}")

    n = end_idx - start_idx
    frame_dsets = ["rawimg", "rawind", "ut1_unix"]  # datasets corresponding to frames, to copy subsets from

    dst.attrs.update(src.attrs)

    for name in src:
        if name not in frame_dsets:
            src.copy(name, dst)

    for name in frame_dsets:
        # _d meaning dataset
        src_d = src[name]

        out_d = dst.create_dataset(
            name,
            shape=(n, *src_d.shape[1:]),
            dtype=src_d.dtype,
            chunks=(min(src_d.chunks[0], n), *src_d.chunks[1:]),
            compression=src_d.compression,
            shuffle=src_d.shuffle,
            compression_opts=src_d.compression_opts,
            fletcher32=src_d.fletcher32,
        )

        step = 256  # copy 256 frames at once
        for i in tqdm(range(start_idx, end_idx, step), desc=name, unit="Frame"):
            j = min(i + step, end_idx)
            out_d[i - start_idx : j - start_idx] = src_d[i:j]

        for key, value in src_d.attrs.items():
            out_d.attrs[key] = value


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
    start_time, end_time = get_start_end_time(args.filename, args.start_time, args.end_time)

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

        copy_datasets(src_f, out_f, start_idx=start_idx, end_idx=end_idx)


if __name__ == "__main__":
    main()
