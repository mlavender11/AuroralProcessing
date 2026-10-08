"""
Find HiST HDF5 files on the data drive by date and camera.

Work in progress. Files are expected to be stored as
``DRIVE_DIR/<YYYYMMDD>/<file>.h5``, with one file per camera.

TODO: this module does not currently import, because of the unfinished
``if ...`` line in ``available_cams``.
"""

from pathlib import Path
from datetime import datetime, timezone
from .datetime_utils import get_start_end_datetime, get_date_str
from .io import available_time_range

DRIVE_DIR = Path("...")  # TODO: placeholder; set to the root of the HDF5 data drive


def get_folder(date: datetime):
    """
    Get the data folder for a given date.

    Parameters
    ----------
    date : datetime.datetime
        Date to look up. Only the year, month and day are used.

    Returns
    -------
    pathlib.Path
        ``DRIVE_DIR / "YYYYMMDD"``. The folder is not checked to exist.
    """
    date_str = date.strftime("%Y%m%d")  # date string to match file
    date_folder = DRIVE_DIR / date_str
    if not date_folder.is_dir():
        raise ValueError(f"Folder for date: {date_str} does not exist. {date_folder}")
    return date_folder


def get_ser_from_fn(fn):
    """
    Get the camera serial number from a HiST file name.

    Not yet implemented; currently returns None.

    Parameters
    ----------
    fn : str or pathlib.Path
        HDF5 file name, e.g. ``2013-03-27-CamSer7196.h5``.

    Returns
    -------
    str
        Camera serial (e.g. ``7196``).
    """
    ...  # TODO Implement extract serial from fn


def available_cams(date: datetime):
    """
    List the cameras that have data for a given date.

    Not yet finished.

    Parameters
    ----------
    date : datetime.datetime
        Date to look up.

    Returns
    -------
    TODO
        Presumably the list of camera serials found in the date's folder.
        TODO: nothing is returned yet.

    Raises
    ------
    ValueError
        If the date's folder contains no ``.h5`` files.
    """
    folder = get_folder(date)
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix == ".h5"]

    if not files:
        raise ValueError(f"No h5 files for date: {date}")

    sers = []
    if ...:
        ...  # TODO implement get sers


def get_available_times(date: datetime, ser=None):
    """
    Print the recorded time range of each camera's file for a given date.

    Not yet finished.

    Parameters
    ----------
    date : datetime.datetime
        Date to look up.
    ser : optional
        Camera serial to restrict the output to. If None, all ``.h5``
        files in the date's folder are listed. TODO: str or int, to match
        ``get_ser_from_fn``. Selecting by serial is not implemented yet.

    Returns
    -------
    None
        Results are printed: camera serial, then start and end times (UTC).

    Raises
    ------
    ValueError
        If ``ser`` is given and no files match it.

    Notes
    -----
    TODO: if ``ser`` is None and the folder has no ``.h5`` files, nothing is
    printed and no error is raised, unlike ``available_cams``.
    """
    folder = get_folder(date)

    if ser is None:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix == ".h5"]
    else:
        files = ...  # TODO implemnt file given date and ser
        if not files:
            raise ValueError(f"No h5 files for camera: {ser} on date: {date}")

    for file in files:
        ser = get_ser_from_fn(file)
        start, end = available_time_range(file)
        print(f"Cam {ser}:")
        print(f"start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"end: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        print()  # TODO check


def extract():
    """
    Not yet implemented.

    TODO: document once implemented; the intended inputs and return value
    can't be determined from the code (possibly a wrapper around
    ``io.extract_clip`` that looks up the file by date and camera).
    """
    ...
