from datetime import datetime, timezone
import h5py
from tqdm.auto import tqdm
from pathlib import Path
from auroral_processing.hdf_utils import get_start_end_idx

FRAME_DSETS = ["rawimg", "rawind", "ut1_unix"]
TIME_DSET = "ut1_unix"


def available_time_range(hdf_fn):
    with h5py.File(hdf_fn, "r") as f:
        ut = f["ut1_unix"]
        start_timestamp = ut[0]
        end_timestamp = ut[-1]

        start_datetime = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
        end_datetime = datetime.fromtimestamp(end_timestamp, tz=timezone.utc)

    return start_datetime, end_datetime


def copy_datasets(src: h5py.File, dst: h5py.File, start_idx, end_idx):
    # Copys HiST HDF5 datasets from start_idx to end_idx
    # expects opened HDF5 files

    if end_idx < start_idx:
        raise ValueError(f"Empty frame range: start_idx={start_idx}, end_idx={end_idx}")

    n = end_idx - start_idx
    frame_dsets = ["rawimg", "rawind", "ut1_unix"]  # datasets corresponding to frames, to copy subsets from

    dst.attrs.update(src.attrs)

    for name in src:
        if name not in frame_dsets:
            src.copy(name, dst)

    for name in frame_dsets:
        # _d meaning dataset
        src_d = src[name]

        out_d = dst.create_dataset(
            name,
            shape=(n, *src_d.shape[1:]),
            dtype=src_d.dtype,
            chunks=(min(src_d.chunks[0], n), *src_d.chunks[1:]),
            compression=src_d.compression,
            shuffle=src_d.shuffle,
            compression_opts=src_d.compression_opts,
            fletcher32=src_d.fletcher32,
        )

        step = 256  # copy 256 frames at once
        for i in tqdm(range(start_idx, end_idx, step), desc=name, unit="Frame"):
            j = min(i + step, end_idx)
            out_d[i - start_idx : j - start_idx] = src_d[i:j]

        for key, value in src_d.attrs.items():
            out_d.attrs[key] = value


def extract_clip(src_fn, start: datetime, end: datetime, out_fn=None):
    if out_fn is None:
        src_fn = Path(src_fn)
        duration_str = f"{start:%H%M%S}-{end:%H%M%S}"
        out_fn = src_fn.with_name(f"{src_fn.stem}_{duration_str}.h5")

    with h5py.File(src_fn, "r") as src_f, h5py.File(out_fn, "w") as out_f:
        start_idx, end_idx = get_start_end_idx(start_time=start, end_time=end, unix_list=src_f["ut1_unix"])
        copy_datasets(src_f, out_f, start_idx=start_idx, end_idx=end_idx)

    return out_fn
