import importlib
from auroral_processing import hdf_utils
from pathlib import Path
import os
from tqdm.auto import tqdm
import datetime
import sys
import h5py
import traceback

# sys.path.append('/Users/michaellavender/Documents/BUSPC/batch_processing')
from auroral_processing.consumers import keogram_hourly as kch
from auroral_processing.consumers import keogram as keogram_consumer
from auroral_processing.consumers import stats as stats_consumer
from auroral_processing.consumers import video as video_consumer

importlib.reload(kch)
importlib.reload(keogram_consumer)
importlib.reload(stats_consumer)
importlib.reload(video_consumer)
importlib.reload(hdf_utils)


h5_parent_folder = Path("/Volumes/Elements/DMC_output")
out_dir_parent = Path("/Users/labb/mlavender/DMC_videos")
out_dir_parent.mkdir(parents=True, exist_ok=True)
h5_files = [Path(f) for f in h5_parent_folder.iterdir() if f.suffix == ".h5"]


for f in h5_files:
    print(f"processing: {str(f.name)}")
    try:
        out_dir = out_dir_parent / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        hdf_utils.make_hourly_videos_keograms(
            hdf_path=f, out_dir=out_dir, bin_size=5, video_quality=10, playback_speed=15, sample_rate_hz=1, font_size=8
        )
    except Exception as e:
        print(f"file {str(f)} failed with error {e}")
        traceback.print_exc()
