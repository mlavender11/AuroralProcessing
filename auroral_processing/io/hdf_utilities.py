import h5py
from tqdm.auto import tqdm
from pathlib import Path
import datetime
from auroral_processing.utils.timing import get_start_end_idx

# TODO Repack, explore
FRAME_DSETS = ["rawimg", "rawind", "ut1_unix"]
TIME_DSET = "ut1_unix"

"""
Read and subset HiST HDF5 camera files.

HiST HDF5 files hold per-frame datasets that share a leading frame axis:

- ``rawimg`` : image frames, shape (n_frames, height, width)
- ``rawind`` : frame indices, shape (n_frames,)
- ``ut1_unix`` : frame timestamps as Unix epoch seconds (UTC), shape (n_frames,)

Any other datasets in the file are treated as metadata and copied whole.
"""


# TODO Rename to copy HiST datasets?
def copy_datasets(src: h5py.File, dst: h5py.File, start_idx, end_idx):
    """
    Copy a range of frames from one open HiST HDF5 file to another.

    The per-frame datasets (``rawimg``, ``rawind``, ``ut1_unix``) are sliced
    to frames ``[start_idx, end_idx)``. Every other dataset is copied
    whole, and both file-level and per-dataset attributes are copied.

    Parameters
    ----------
    src : h5py.File
        Open source file (read mode).
    dst : h5py.File
        Open destination file (write mode). It must not already contain
        datasets with the same names.
    start_idx : int
        Index of the first frame to copy (inclusive).
    end_idx : int
        Index to stop at (exclusive).

    Raises
    ------
    ValueError
        If ``end_idx < start_idx``.

    Notes
    -----
    Each output dataset keeps the source's chunking (with the frame-axis
    chunk capped at the number of frames copied), compression, shuffle and
    fletcher32 settings. Frames are copied 256 at a time to limit memory use.

    TODO: ``start_idx == end_idx`` passes the check above but creates
    zero-length datasets, and ``min(src_d.chunks[0], 0)`` gives a zero
    chunk size, which h5py may reject. Confirm what should happen here.
    TODO: assumes each per-frame dataset is chunked (``src_d.chunks`` is not
    None); a contiguous dataset would raise a TypeError. Confirm all HiST
    files are chunked.
    """
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
    """
    Save the frames between two times from a HiST HDF5 file to a new file.

    Parameters
    ----------
    src_fn : str or pathlib.Path
        Path to the source HDF5 file.
    start : datetime.datetime
        UTC start time of the clip.
    end : datetime.datetime
        UTC end time of the clip.
    out_fn : str or pathlib.Path, optional
        Output path. By default the file is written next to ``src_fn`` and
        named ``<stem>_<HHMMSS>-<HHMMSS>.h5`` from the start and end times.
        An existing file at this path is overwritten.

    Returns
    -------
    str or pathlib.Path
        Path of the written file.

    Notes
    -----
    ``start`` and ``end`` are matched to the nearest frames using
    ``hdf_utils.get_start_end_idx``. Because ``copy_datasets`` treats the
    end index as exclusive, the frame nearest ``end`` is not included.
    TODO: confirm whether the end frame should be included.
    """
    if out_fn is None:
        src_fn = Path(src_fn)
        duration_str = f"{start:%H%M%S}-{end:%H%M%S}"
        out_fn = src_fn.with_name(f"{src_fn.stem}_{duration_str}.h5")

    with h5py.File(src_fn, "r") as src_f, h5py.File(out_fn, "w") as out_f:
        start_idx, end_idx = get_start_end_idx(start_time=start, end_time=end, unix_list=src_f["ut1_unix"])
        copy_datasets(src_f, out_f, start_idx=start_idx, end_idx=end_idx)

    return out_fn
