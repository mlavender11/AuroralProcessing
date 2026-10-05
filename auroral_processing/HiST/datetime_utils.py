"""
Parse dates and times from HiST file names.

HiST HDF5 files are expected to have names containing a ``YYYY-MM-DD``
date, e.g. ``2013-03-27-CamSer7196.h5``.
"""

from datetime import datetime, timezone
from pathlib import Path
import re


def get_date_str(filename):
    """
    Extract the ``YYYY-MM-DD`` date from a file name.

    Parameters
    ----------
    filename : str
        File name such as ``2013-03-27-CamSer7196.h5``. The first match
        anywhere in the string is used, not only at the start.

    Returns
    -------
    str
        The date in ``YYYY-MM-DD`` form.

    Raises
    ------
    ValueError
        If no ``YYYY-MM-DD`` pattern is found.

    Notes
    -----
    Only the digit pattern is checked, so an invalid date such as
    ``2013-13-45`` is still returned.
    """
    # Get the date as a string
    # Expects file name like 2013-03-27-CamSer7196.h5 - just needs to start with YYYY-MM-DD

    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)

    if not match:
        raise ValueError(f"Filename {filename} does not contain a valid date")
    else:
        return match.group(0)


def get_start_end_datetime(filename, start_time, end_time):
    """
    Combine the date in a file name with start and end times of day.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path whose base name contains a ``YYYY-MM-DD`` date, e.g.
        ``2013-03-27-CamSer7196.h5``.
    start_time : str
        UTC start time of day, ``HH:MM:SS``.
    end_time : str
        UTC end time of day, ``HH:MM:SS``.

    Returns
    -------
    start : datetime.datetime
        Timezone-aware UTC datetime for the start time on the file's date.
    end : datetime.datetime
        Timezone-aware UTC datetime for the end time on the file's date.

    Raises
    ------
    ValueError
        If the file name has no date, or a time is not ``HH:MM:SS``.

    Notes
    -----
    Both times are put on the same date, so a range that crosses midnight
    UTC gives ``end`` earlier than ``start``.
    TODO: confirm whether ranges crossing midnight UTC need handling.
    """
    # Start and end datetime objects in UTC
    # Expects file name like 2013-03-27-CamSer7196.h5 - just needs to start with YYYY-MM-DD
    # start_time and end_time in HH:MM:SS

    # DATE AND START TIME
    tz = timezone.utc
    date_str = get_date_str(Path(filename).name)
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Start and end time from strings
    start_time = datetime.strptime(start_time, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time, "%H:%M:%S").time()

    # Timezone aware start and end timestamps in UTC format (with date)
    start = datetime.combine(date, start_time, tzinfo=tz)
    end = datetime.combine(date, end_time, tzinfo=tz)

    return start, end


def get_time_bounds():
    """
    Not yet implemented.

    TODO: document once implemented; the intended inputs and return value
    can't be determined from the code.
    """
    ...
