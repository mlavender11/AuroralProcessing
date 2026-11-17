import numpy as np
from tqdm.auto import tqdm


# driver - used to pull from the end of a chain of generator stages
def write_h5(
    src,
    hdf_file,
    dataset_name,
    total_frames=None,
    time_name=None,
    dtype=None,
    compression="gzip",
    compression_opts=1,
    shuffle=True,
    overwrite=False,
    progress=True,
):
    """
    Drain a chain of (t, frame) stages into an image dataset plus a matching
    time dataset.

    Parameters
    ----------
    src : iterable of (t, frame)
        the last stage of the chain
    hdf_file : h5py.File
        open file, writable
    dataset_name : str
        image dataset to write, e.g. "v4"
    total_frames : int, optional
        expected number of output frames. Only used to pre-size the datasets
        and for the progress bar; the datasets grow or shrink to the real count.
    time_name : str, optional
        time dataset to write, by default f"{dataset_name}_ut1_unix"
    dtype : numpy dtype, optional
        dtype of the image dataset. By default the dtype of the first frame, so a chain that
        ends in float frames is stored as float (end the chain in to_uint16 for uint16).
    overwrite : bool
        replace existing datasets of the same names. Otherwise an existing dataset is an error.
    progress : bool
        show a progress bar

    Returns
    -------
    int : number of frames written

    The two datasets point at each other through the attributes
    "time_dataset" (on the images) and "frames_dataset" (on the times).
    """
    f = hdf_file
    if time_name is None:
        time_name = f"{dataset_name}_ut1_unix"

    for name in (dataset_name, time_name):
        if name in f:
            if not overwrite:
                raise FileExistsError(f"dataset {name!r} already exists in {f.filename}; pass overwrite=True to replace it")
            del f[name]

    imgs = times = None
    i = 0

    for t, frame in tqdm(src, total=total_frames, desc=dataset_name, unit="frame", disable=not progress):
        if imgs is None:
            frame_shape = frame.shape
            initial_len = total_frames or 1024
            imgs = f.create_dataset(
                dataset_name,
                shape=(initial_len,) + frame_shape,
                maxshape=(None,) + frame_shape,  # resizable
                dtype=frame.dtype if dtype is None else dtype,
                chunks=(1,) + frame_shape,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )
            times = f.create_dataset(
                time_name,
                shape=(initial_len,),
                maxshape=(None,),
                dtype=np.float64,
            )

        if i == imgs.shape[0]:
            new_size = 2 * i
            imgs.resize(new_size, axis=0)
            times.resize(new_size, axis=0)

        imgs[i] = frame
        times[i] = t
        i += 1

    # trim to the frames actually written
    if imgs is not None and imgs.shape[0] != i:
        imgs.resize(i, axis=0)
        times.resize(i, axis=0)

    return i


def read_h5(frame_dataset, time_dataset=None, block=64):
    """
    Read an HDF5 image dataset frame by frame, paired with its timestamps.

    Parameters
    ----------
    frame_dataset : h5py.Dataset
        frames, shape (n_frames, ny, nx)
    time_dataset : h5py.Dataset or array-like, optional
        one timestamp per frame (e.g. ut1_unix). If None, t is the frame index.
    block : int, optional
        number of frames to read from disk at once, by default 64

    Yields
    ------
    (t, frame) : (float, np.ndarray)
    """
    n = len(frame_dataset)
    if time_dataset is not None and len(time_dataset) != n:
        raise ValueError(
            f"time dataset {time_dataset.name} has {len(time_dataset)} entries, frame dataset {frame_dataset.name} has {n}"
        )

    for start in range(0, n, block):
        # pull a block at a time, for memory performance
        chunk = frame_dataset[start : start + block]
        if time_dataset is not None:
            tchunk = np.asarray(time_dataset[start : start + block], dtype=np.float64)
        else:
            tchunk = np.arange(start, start + len(chunk), dtype=np.float64)
        for t, frame in zip(tchunk, chunk):
            yield t, frame
