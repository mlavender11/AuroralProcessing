import argparse
from auroral_processing.io.database import get_available


def available_dates(): ...


def datetime_from_str(date_str): ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore and extract from HiST database")
    parser.add_argument("date", default=None, help="Target date, YYYY-MM-DD")
    parser.add_argument("-s", "--start_time", default=None, help="Target start time, HH:MM:SS")
    parser.add_argument("-e", "--end_time", default=None, help="Target end time, HH:MM:SS")
    parser.add_argument("-c", "--camera_serial", default=None, help="Target camera serial, e.x. 7196")
    parser.add_argument(
        "-o", "--output_folder", default=None, help="Extraction output folder, defaults to same as source"
    )
    parser.add_argument(
        "-a",
        "--available",
        default=False,
        action="store_true",
        help="Gives available times, dates, and cameras, defaults to false",
    )
    parser.add_argument("-e", "--extract", default=False, action="store_true", help="Extract file, defaults to false")
    args = parser.parse_args()

    if args.available and args.extract:
        raise ValueError("Error: both -a and -e given, choose one or the others")
    elif args.available:
        get_available(...)
    elif args.extract:
        extract(...)  # TODO implement, in io?
    else:
        raise ValueError("Error: neither -a or -e selected")
