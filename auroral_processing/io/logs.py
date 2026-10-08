from pathlib import Path
import datetime


def parse_log_for_start_time(log_file: Path | str) -> datetime.datetime:
    import re

    log_file = Path(log_file)
    if not log_file.is_file():
        raise FileNotFoundError(log_file)

    with log_file.open("r") as f:
        for line in f:
            if "Acquisition started" in line:
                break
        else:
            raise ValueError(f"No 'Acquisition started' line found in {log_file}")

    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)", line)
    if match:
        start_time = datetime.datetime.strptime(match.group(), "%Y-%m-%dT%H:%M:%S.%f")
    else:
        raise ValueError(f"no start time found in {line.strip()}")

    return start_time
