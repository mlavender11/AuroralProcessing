from pathlib import Path
from datetime import datetime, timezone
import re


def build_output_paths(hdf_path: Path, out_dir: Path, unix_time, i: int, *, camera, serial):
    """
    build output paths like CamSer1387_20130414_07.png

    Parameters
    ----------
    hdf_path : Path
        hdf file path
    out_dir : Path
        output director
    unix_time : _type_
        time at beginning of video/keogram subset

    Returns
    -------
    _type_
        video path (stem.mp4), keogram path (stem.png)
    """

    # cam_str = re.search(r"CamSer\d+", hdf_path.stem).group()

    dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    date_str = dt.strftime("%Y%m%d")
    hour_str = dt.strftime("%H")

    # fn = out_dir / f"{cam_str}_{date_str}_{hour_str}"
    # fn = out_dir / f"{cam_str}_{date_str}_{i}"
    # fn = out_dir / f"DMC_{date_str}_{hour_str}"
    # fn = out_dir / f'{camera}-'

    if camera == "HST":
        fn = out_dir / f"{camera}-{serial}-{date_str}-{i}"
    else:
        fn = out_dir / f"{camera}-{date_str}-{i}"

    return fn.with_suffix(".mp4"), fn.with_suffix(".png")
