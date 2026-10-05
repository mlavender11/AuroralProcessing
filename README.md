# AuroralProcessing
![status: restructuring](https://img.shields.io/badge/status-restructuring-orange)

Tools for processing high-speed auroral camera data (HiST / DMC): converting raw
camera binaries to HDF5, cutting clips, and producing videos, keograms and
summary statistics. Also contains exploratory optical-flow work on auroral motion.

> [!WARNING]
> I am currently restructuring this package. The planned layout is on the
> [`restructuring`](../../tree/restructuring) branch; `main` holds the current working code.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/mlavender11/AuroralProcessing.git
cd AuroralProcessing
pip install -e .
```

Optional, depending on what you run:

- `histutils` – reading raw `.DMCdata` files
- `pyaurorax` – downloading THEMIS all-sky data for external keograms
- `opencv-python`, `openpiv` – optical-flow experiments

## Layout

```
auroral_processing/     Python package
    binary_to_hdf.py      raw .DMCdata (+ .xml, .nmea) -> HDF5
    hdf_utils.py          normalisation, videos, keograms (main functionality)
    consumers/            per-frame endpoints with update() / finalize():
                          video, keogram, hourly keogram, stats
    hist/                 HiST HDF5 helpers
        io.py               time range of a file, clip extraction
        datetime_utils.py   date/time parsing for filenames and clips
        database.py         find files by date and camera (work in progress)
scripts/                Command-line and batch scripts
notebooks/              DMC_Processing and FullPipeline notebooks
optical_flow/           Optical-flow experiments: OpenCV (Farneback, TV-L1),
                        OpenPIV, FLCT, MATLAB (Black & Anandan / Horn-Schunck),
                        and reference papers
```

## Data format

HDF5 files are expected to contain image frames in `rawimg` and per-frame UT
timestamps (Unix seconds) in `ut1_unix`.

## Usage

Make a video from an HDF5 file:

```bash
python scripts/make_video.py path/to/file.h5 -o path/to/output.mp4
```

Extract a time window from an HDF5 file:

```bash
python scripts/extract_hdf_subset.py path/to/file.h5 07:30:00 08:00:00 -o clip.h5
```

Other scripts:

| Script | Purpose |
| --- | --- |
| `DMC_Processing.py` | Batch-convert folders of `.DMCdata` on a drive to HDF5 |
| `dmc_video_maker.py` | Hourly videos and keograms for every HDF5 file in a folder |
| `process_from_spreadsheet.py` | Process the nights listed in a spreadsheet |
| `make_external_kv.py` | Keograms from THEMIS all-sky imagers (e.g. Fort Yukon) |
| `database_access.py` | Look up available dates/cameras on the data drive (unfinished) |

Several scripts have hard-coded paths (e.g. `/Volumes/Elements/...`); edit these
for your machine before running.

## Notes

Generated outputs (`*.h5`, `*.mp4`, `*.png`, `*.npz`, `*.csv`, `*.xlsx`, `*.log`)
are git-ignored.
