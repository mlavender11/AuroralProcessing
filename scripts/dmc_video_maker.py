from pathlib import Path
import traceback

from auroral_processing.products.hourly import make_hourly_videos_keograms


h5_parent_folder = Path("/Volumes/Elements/DMC_output")
out_dir_parent = Path("/Users/labb/mlavender/DMC_videos")
out_dir_parent.mkdir(parents=True, exist_ok=True)
h5_files = [Path(f) for f in h5_parent_folder.iterdir() if f.suffix == ".h5"]


for f in h5_files:
    print(f"processing: {str(f.name)}")
    try:
        out_dir = out_dir_parent / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        make_hourly_videos_keograms(
            hdf_path=f,
            out_dir=out_dir,
            camera="DMC",
            bin_size=5,
            video_quality=10,
            playback_speed=15,
            sample_rate_hz=1,
            font_size=8,
        )
    except Exception as e:
        print(f"file {str(f)} failed with error {e}")
        traceback.print_exc()
