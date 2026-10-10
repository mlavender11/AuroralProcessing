import h5py
import numpy as np
from tqdm.auto import tqdm


# driver - used to pull from the end of a chain of generator stages
def write_h5(src, hdf_file, dataset_name, total_frames, dtype=np.uint16, compression=None, block=64, attrs=None):
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
            )
            for key, val in (attrs or {}).items():
                dataset.attrs[key] = val  # what is this?

        dataset[i] = frame
        i += 1

    if dataset is not None and dataset.shape[0] != i:
        dataset.resize(i, axis=0)

    return i
