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
    total_frames,
    dtype=np.uint16,
    compression="gzip",
    compression_opts=1,
    shuffle=True,
    block=64,
    image_attrs=True,
):
    f = hdf_file

    if dataset_name in f:
        del f[dataset_name]  # TODO implement protections
        # raise ValueError("Can't overwrite a dataset")

    dataset = None
    i = 0
    for frame in tqdm(src, total=total_frames):
        if dataset is None:
            frame_shape = frame.shape
            dataset = f.create_dataset(
                dataset_name,
                shape=(total_frames,) + frame_shape,
                dtype=dtype,
                chunks=(1,) + frame_shape,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )
            if image_attrs:
                for key, val in (IMAGE_ATTRS or {}).items():
                    dataset.attrs[key] = val  # what is this?

        dataset[i] = frame
        i += 1

    if dataset is not None and dataset.shape[0] != i:
        dataset.resize(i, axis=0)

    return i
