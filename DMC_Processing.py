#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import hdf_utils
from tqdm.auto import tqdm
from datetime import datetime

# sys.path.append('/Users/michaellavender/Documents/BUSPC/batch_processing')
# from binary_to_hdf import whole_binary_to_hdf


def start_time_from_fn(fn: Path):
    dt = datetime.strptime(fn.stem, "%Y-%m-%dT%H-%M-%S")
    return dt


def convert(drive_path, out_dir, excluded_terms):
    from histutils.convert.__main__ import convert_DMC_to_hdf5

    drive_path = Path(drive_path)
    DMC_folders = []
    for item in drive_path.iterdir():
        try:
            if item.is_dir():
                if any(item.glob("*.DMCdata")):
                    DMC_folders.append(item)
        except Exception as e:
            print(f"Skipped {item.name}: {e}")

    DMC_files = []
    for folder in DMC_folders:
        try:
            for DMC_file in folder.glob("*.DMCdata"):
                DMC_files.append(DMC_file)
        except Exception as e:
            print(f"Skipped folder {folder.name}: {e}")

    DMC_files_cleaned = [file for file in DMC_files if not any(term in file.name for term in excluded_terms)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for DMC_fn in DMC_files_cleaned:
        print(f"Converting {DMC_fn}")
        try:
            out_fn = out_dir / DMC_fn.with_suffix(".h5").name
            start_time = start_time_from_fn(DMC_fn)
            print(f"start time: {start_time}")

            convert_DMC_to_hdf5(DMC_fn, out_fn, {"header_bytes": 4, "startUTC": start_time})
            print("." * 40 + "\n")

        except Exception as e:
            print(f"Error in file {DMC_fn}: {e}")


def make_summary(h5_folder, out_folder):
    import hdf_utils
    from pathlib import Path
    from tqdm.auto import tqdm
    import traceback

    h5_parent_folder = Path(h5_folder)
    out_dir_parent = Path(out_folder)
    out_dir_parent.mkdir(parents=True, exist_ok=True)
    h5_files = [Path(f) for f in h5_parent_folder.iterdir() if f.suffix == ".h5"]

    for f in h5_files:
        print(f"processing: {str(f.name)}")
        try:
            out_dir = out_dir_parent / f.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            hdf_utils.make_hourly_videos_keograms(
                hdf_path=f,
                out_dir=out_dir,
                bin_size=5,
                video_quality=10,
                playback_speed=15,
                sample_rate_hz=1,
                font_size=8,
            )
        except Exception as e:
            print(f"file {str(f)} failed with error {e}")
            traceback.print_exc()


def main():
    data_path = Path("/Volumes/L")
    h5_folder = Path("/Volumes/Elements/DMC_output/813")
    video_folder = Path("/Users/labb/mlavender/DMC_videos/813")
    excluded_terms = ["frames", "test"]

    convert(drive_path=data_path, out_dir=h5_folder, excluded_terms=excluded_terms)
    make_summary(h5_folder=h5_folder, out_folder=video_folder)


if __name__ == "__main__":
    main()
