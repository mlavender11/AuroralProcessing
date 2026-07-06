"""
config.py -- everything you need to EDIT lives here.

Fill in the four sections marked  >>> EDIT <<< . Nothing else in the project
should need changing to get a first run going. Read the comments; they explain
the assumptions each section is making about your data layout.
"""

from pathlib import Path

# TODO setup
# >>> EDIT 1: where your data drives are mounted ------------------------------
# One entry per PHYSICAL drive. The 'id' is just a label you choose.
# 'root' is the folder under which that drive's nightly data lives.
# This list is also what controls parallelism: the runner starts exactly one
# worker per entry here, so one worker per disk.
DRIVES = [
    {"id": "drive1", "root": Path("/mnt/aurora1")},
    {"id": "drive2", "root": Path("/mnt/aurora2")},
    # add as many as you physically have...
]

# TODO setup
# >>> EDIT 2: where outputs and scratch go ------------------------------------
HDF_DIR = Path(
    "/mnt/scratch/aurora_hdf"
)  # transient and persistent HDF; fast local disk if possible TODO update process night to not move hdf files
OUTPUT_DIR = Path("/mnt/aurora_outputs")  # keograms + videos (small, keep these)
DB_PATH = Path("/mnt/aurora_outputs/catalog.db")

# How the converted HDF stores its frames.
HDF_DATASET = "rawimg"  # the (frames, height, width) image dataset
HDF_TIME = "ut1_unix"  # the per-frame UTC timestamp dataset

# What to do with each night's HDF after the keogram + video are made:
#   "none"        -- delete every HDF (smallest footprint; re-convert if needed later)
#   "all"         -- keep every HDF (cumulative storage = nights x HDF size; can be huge)
#   "interesting" -- keep only nights whose 'motion' stat is high enough (recommended)
KEEP_HDF = "all"
KEEP_MOTION_THRESHOLD = 2.0  # used only when KEEP_HDF == "interesting"; tune from triage numbers

# Video parameters
PLAYBACK_SPEED = 4
BIN_SIZE = 5
# VIDEO_QUALITY = 8
VIDEO_QUALITY = 8  # testing only TODO remove

# Keogram parameters EACH FRAME IF NONE
BIN_WIDTH_SECONDS = 60

CMAP = "gray"


# TODO fix this
# >>> EDIT 3: how to find the nights on each drive ----------------------------
def discover_nights(drive):
    """Return a list of (night_id, raw_path) for one drive.

    A 'night' here is assumed to be one subfolder named like a date, e.g.
    /mnt/aurora1/20240115/ . The night_id is just that folder name and becomes
    the unique key for the whole pipeline, so it must be unique across drives
    (a date usually is, for a single camera). Adapt the glob if your layout
    differs -- e.g. if each night is a single large file rather than a folder.
    """
    nights = []
    for path in sorted(drive["root"].glob("[0-9]" * 8)):  # 8-digit names like 20240115
        if path.is_dir():
            nights.append((path.name, path))  # (night_id, raw_path)
    return nights


# >>> EDIT 4: wrap your three existing Python functions -----------------------
# Import your real functions and call them inside these thin wrappers. The
# wrapper is just an ADAPTER: the rest of the pipeline always calls
# convert_to_hdf(raw_path, hdf_out) etc., and inside each wrapper you translate
# those two arguments into whatever your function's real signature expects.
# Keep these wrapper names exactly as they are; only change the bodies.
#
# Headless machines: if your keogram/video functions use matplotlib, force a
# non-interactive backend ONCE, at import time, or they may error in a worker
# with no display attached:
import matplotlib

matplotlib.use("Agg")
