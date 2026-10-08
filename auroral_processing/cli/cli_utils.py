import argparse
from datetime import datetime, timezone


def hhmmss(s, *, tz=timezone.utc):
    """
    Used with argparser, get date as HH:MM:SS and return time object
    Defaults to UTC timezone
    """
    try:
        dt = datetime.strptime(s, "%H:%M:%S")
        tz_dt = dt.replace(tzinfo=tz)
        return tz_dt.timetz()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r} is not HH:MM:SS")
