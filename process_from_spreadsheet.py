import pandas as pd
from hdf_utils import compute_norm_from_hdf, make_hourly_videos_keograms
import datetime
from pathlib import Path
from tqdm.auto import tqdm
import sys


def resolve_drive(drive_key):
    volumes_dir = Path("/Volumes")
    matches = [v for v in volumes_dir.iterdir() if drive_key.lower() in v.name.lower()]

    if not matches:
        raise FileNotFoundError(f"no mounted volume matching {drive_key!r} found in /Volumes")
    if len(matches) > 1:
        raise ValueError(f"multiple volumes match {drive_key!r}: {matches}")

    return matches[0]


def to_date(value):
    """Coerce whatever Excel/pandas handed back into a datetime.date."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.datetime.strptime(value.strip(), "%Y-%m-%d").date()
    raise ValueError(f"unrecognized date value: {value!r} ({type(value)})")


def to_time(value):
    """Coerce whatever Excel/pandas handed back into a datetime.time."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.time()
    if isinstance(value, datetime.datetime):
        return value.time()
    if isinstance(value, datetime.time):
        return value
    if isinstance(value, str):
        return datetime.datetime.strptime(value.strip(), "%H:%M:%S").time()

    raise ValueError(f"unrecognized time value: {value!r} ({type(value)})")


def to_cam_ser(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_paths(drive_path, date, cam_ser):
    date_str = date.strftime("%Y-%m-%d")
    hdf_folder = Path(drive_path) / "HDF" / date_str / cam_ser
    h5_files = list(hdf_folder.glob("*.h5"))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"no .h5 file found in {hdf_folder}")
    if len(h5_files) > 1:
        raise Exception(f"more than one .h5 file found in {hdf_folder}: {h5_files}")

    hdf_path = h5_files[0]
    out_dir = hdf_folder / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    return hdf_path, out_dir


def process_sheet(sheet_fn, drive_path):
    df = pd.read_excel(sheet_fn)
    if "processed" not in df.columns:
        df["processed"] = False
    if "error" not in df.columns:
        df["error"] = ""

    df["processed"] = df["processed"].fillna(False).astype(bool)
    df["error"] = df["error"].fillna("")

    to_run = df[~df["processed"]]
    print(f"{len(to_run)} of {len(df)} rows not yet processed")

    for idx, row in tqdm(to_run.iterrows(), total=len(to_run)):
        try:
            date = to_date(row["date"])
            start_time = to_time(row["start_time"])
            cam_ser = to_cam_ser(row["cam_ser"])

            hdf_path, out_dir = build_paths(drive_path=drive_path, date=date, cam_ser=cam_ser)
            if start_time is not None:
                start_dt = datetime.datetime.combine(date, start_time, tzinfo=datetime.timezone.utc)
            else:
                start_dt = None
            run_hdf(hdf_path, sample_interval_seconds=1, start_time=start_dt, out_dir=out_dir)

            df.at[idx, "processed"] = True
            df.at[idx, "error"] = ""

        except Exception as e:
            df.at[idx, "processed"] = False
            df.at[idx, "error"] = str(e)
            tqdm.write(f"  {row.get('date')} / {row.get('cam_ser')} failed: {e}")

    df.to_excel(sheet_fn, index=False)
    print(f"done. wrote results back to {sheet_fn}")


def run_hdf(hdf_fn, sample_interval_seconds, start_time, out_dir):
    norm = compute_norm_from_hdf(hdf_fn=hdf_fn, sample_interval_seconds=sample_interval_seconds, start_time=start_time)
    make_hourly_videos_keograms(hdf_path=hdf_fn, out_dir=out_dir, bin_size=5, playback_speed=15, output_hz=1, norm=norm)


if __name__ == "__main__":
    sheet = sys.argv[1]
    drive_name = sys.argv[2]
    drive_path = resolve_drive(drive_name)
    process_sheet(sheet, drive_path)
