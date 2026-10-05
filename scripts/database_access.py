from pathlib import Path
import argparse
import h5py
from datetime import datetime

DRIVE_DIR = Path("...")


def available_dates(): ...


def available_times(date, camera): ...


def available_cams(date): ...


def get_available(): ...


def extract(): ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore and extract from HiST database")
    parser.add_argument("data", default=None, help="Target date, YYYY-MM-DD")
    parser.add_argument("-c", "--camera_serial", default=None, help="Target camera serial, e.x. 7196")
    parser.add_argument(
        "-o", "--output_folder", default=None, help="Extraction output folder, defaults to same as source"
    )
    parser.add_argument(
        "-a", "--available", default=False, help="Gives available times, dates, and cameras, defaults to false"
    )
    parser.add_argument("-e", "--extract", default=False, help="Extract file, defaults to false")
    args = parser.parse_args()

    if args.available and args.extract:
        raise ValueError("Error: both -a and -e given, choose one or the others")
    elif args.available:
        get_available(...)
    elif args.extract:
        extract(...)
    else:
        raise ValueError("Error: neither -a or -e selected")
