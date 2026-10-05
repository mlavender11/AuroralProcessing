from pathlib import Path
from datetime import datetime, timezone
from .datetime_utils import get_start_end_datetime, get_date_str
from .io import available_time_range

DRIVE_DIR = Path("...")

def get_folder(date: datetime):
    date_str = date.strftime("%Y%m%d")  # date string to match file
    date_folder = DRIVE_DIR / date_str
    return date_folder

def get_ser_from_fn(fn):
    ... # TODO Implement extract serial from fn

def available_cams(date: datetime):
    folder = get_folder(date)
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix() == ".h5"]

    if not files:
        raise ValueError(f"No h5 files for date: {date}")

    sers = []
    if ... # TODO implement get sers


def get_available_times(date: datetime, ser=None):
    folder = get_folder(date)

    if ser is None:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix() == ".h5"]
    else:
        files = ... # TODO implemnt file given date and ser
        if not files:
            raise ValueError(f'No h5 files for camera: {ser} on date: {date}')

    for file in files:
        ser = get_ser_from_fn(file)
        start, end = available_time_range(file)
        print(f'Cam {ser}:')
        print(f'start: {start.strftime('%Y-%m-%d %H:%M:S')}')
        print(f'end: {end.strftime('%Y-%m-%d %H:%M:S')}')
        print() # TODO check

    

def extract(): ...
