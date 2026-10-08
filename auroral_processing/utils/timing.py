import numpy as np
import datetime


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


def _assert_utc(dt: datetime.datetime) -> None:
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


def get_hourly_boundaries(start_time: datetime.datetime, end_time: datetime.datetime):
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


def get_hourly_sub_idx(ut, start_time: datetime.datetime, end_time: datetime.datetime):
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
