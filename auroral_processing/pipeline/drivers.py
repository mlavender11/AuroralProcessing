import h5py
import numpy as np
from tqdm.auto import tqdm

IMAGE_ATTRS = {
    "CLASS": np.bytes_("IMAGE"),
    "IMAGE_VERSION": np.bytes_("1.2"),
    "IMAGE_SUBCLASS": np.bytes_("IMAGE_GRAYSCALE"),
    "DISPLAY_ORIGIN": np.bytes_("UL"),
    "IMAGE_WHITE_IS_ZERO": np.uint8(0),
}


# driver - used to pull from the end of a chain of generator stages
def write_h5(
    src,
    hdf_file,
    dataset_name,
    total_frames=None,
    time_name=None,
    time_attrs=None,
    dtype=np.uint16,
    compression="gzip",
    compression_opts=1,
    shuffle=True,
    image_attrs=True,
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
    time_attrs : dict, optional
        attributes for the time dataset (e.g. copied from the source time dataset)
    block : int, optional
        frames buffered per write, by default 64

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
            del f[name]  # TODO implement protections

    dataset = times = None
    i = 0

    for t, frame in tqdm(src, total=total_frames):
        if dataset is None:
            frame_shape = frame.shape
            dataset = f.create_dataset(
                dataset_name,
                shape=(total_frames,) + frame_shape,
                maxshape=(None,) + frame_shape,  # resizable
                dtype=dtype,
                chunks=(1,) + frame_shape,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )
            times = f.create_dataset(
                time_name,
                shape=(total_frames,),
                maxshape=(None,),
                dtype=np.float64,
            )
            if image_attrs:
                for key, val in IMAGE_ATTRS.items():
                    dataset.attrs[key] = val
            dataset.attrs["time_dataset"] = time_name
            times.attrs["frames_dataset"] = dataset_name
            for key, val in (time_attrs or {}).items():
                times.attrs[key] = val
        dataset[i] = frame
        i += 1

    # trim to the frames actually written
    if dataset is not None and dataset.shape[0] != i:
        dataset.resize(i, axis=0)
        times.resize(i, axis=0)

    return i
