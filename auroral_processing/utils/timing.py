import numpy as np
from datetime import datetime, timezone
import datetime
import h5py
from pathlib import Path
import re


def find_closest_item(arr, key):
    """
    Finds the index of the element in arr that is closest in value to key

    Parameters
    ----------
    arr : array_like
        Sorted (ascending) 1D array to be searched
    key : float
        Value to find closest element to.

    Returns
    -------
    int
        Index of element in arr that is closest to key.
    """
    idx = np.searchsorted(arr, key)
    if idx == 0:
        return idx
    if idx == len(arr):
        return idx - 1

    left, right = idx - 1, idx
    return left if abs(arr[left] - key) <= abs(arr[right] - key) else right


def find_time_idx(time: datetime, ut_list):
    _assert_utc(time)
    time_unix = int(time.timestamp())
    idx = find_closest_item(ut_list, time_unix)
    return idx


def get_start_end_idx(start_time: datetime, end_time: datetime, unix_list):
    """
    Given a start time, end time, and unix epoch time list, returns index of the start and end time

    Parameters
    ----------
    start_time : datetime.datetime
        UTC start time to locate in unix_list
    end_time : datetime.datetime
        UTC end time to locate in unix_list
    unix_list : array_like
        Sorted (ascending) list of unix epoch time values.

    Returns
    -------
    tuple[int, int]
        Index of the frame closest to start_time and index of the frame
        closest to end_time.
    """
    start_idx = find_time_idx(start_time, unix_list)
    end_idx = find_time_idx(end_time, unix_list)

    # _assert_utc(start_time)
    # _assert_utc(end_time)

    # start_time_unix = int(start_time.timestamp())
    # end_time_unix = int(end_time.timestamp())

    # start_idx = find_closest_item(unix_list, start_time_unix)
    # end_idx = find_closest_item(unix_list, end_time_unix)

    return start_idx, end_idx


def _assert_utc(dt: datetime) -> None:
    """
    Asserts that a datetime object is in UTC.

    Parameters
    ----------
    dt : datetime.datetime
        Datetime object to be checked.

    Raises
    ------
    ValueError
        Raises if dt is not timezone-aware and in UTC.
    """
    if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"datetime must be timezone-aware and in UTC, got: {dt!r}")


def get_hourly_boundaries(start_time: datetime, end_time: datetime):
    """
    Given a start and end time, return a list containing the start time, the top of every hour until the end time, and the end time.

    Parameters
    ----------
    start_time : datetime.datetime
    end_time : datetime.datetime

    Returns
    -------
    list[datetime.datetime]
        List containing start time, end time, and the top of every hour in
        between, in ascending order.
    """

    if start_time.minute > 0 or start_time.second > 0:
        first_whole_hour = start_time.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        hours = [start_time]
    else:
        first_whole_hour = start_time
        hours = []

    current_hour = first_whole_hour

    while current_hour < end_time:
        hours.append(current_hour)
        current_hour += datetime.timedelta(hours=1)

    hours.append(end_time)

    return hours


def get_hourly_sub_idx(ut, start_time: datetime, end_time: datetime):
    """
    Given a start time, end time, and list of unix epoch times, return a list
    containing the indices of the start time, the top of every hour until the
    end time, and the end time.

    Parameters
    ----------
    ut : array_like
        List of unix epoch times.
    start_time : datetime.datetime
    end_time : datetime.datetime

    Returns
    -------
    list[int]
        List containing the indices, in the ut list, of the start time, end
        time, and every hour boundary in between.
    """

    _assert_utc(start_time)
    _assert_utc(end_time)

    hours = get_hourly_boundaries(start_time, end_time)

    hour_idxs = [find_closest_item(ut, hour.timestamp()) for hour in hours]

    return hour_idxs


def available_time_range(hdf_fn):
    """
    Get the time span covered by a HiST HDF5 file.

    Parameters
    ----------
    hdf_fn : str or pathlib.Path
        Path to an HDF5 file containing a ``ut1_unix`` dataset.

    Returns
    -------
    start_datetime : datetime.datetime
        UTC time of the first frame (timezone-aware).
    end_datetime : datetime.datetime
        UTC time of the last frame (timezone-aware).

    Notes
    -----
    Uses the first and last entries of ``ut1_unix``, so it assumes the
    timestamps are sorted in ascending order.
    """
    with h5py.File(hdf_fn, "r") as f:
        ut = f["ut1_unix"]
        start_timestamp = ut[0]
        end_timestamp = ut[-1]

        start_datetime = datetime_from_unix(start_timestamp)
        end_datetime = datetime_from_unix(end_timestamp)

    return start_datetime, end_datetime


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
    TODO Not yet implemented.
    """
    ...


def datetime_from_unix(timestamp: float) -> datetime:
    """
    Convert a Unix timestamp to a timezone-aware UTC datetime

    Parameters
    ----------
    timestamp : float or int
        Seconds since the Unix epoch. Anything convertible with ``float()`` is accepted.

    Returns
    -------
    datetime
        Timezone-aware datetime with ``tzinfo=timezone.utc``.
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
