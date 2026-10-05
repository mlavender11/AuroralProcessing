from datetime import datetime, timezone
from pathlib import Path
import re


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
