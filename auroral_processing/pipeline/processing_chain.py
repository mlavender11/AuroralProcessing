import h5py

# TODO read chain from setup file


def chain(src, *stages):
    # Chain together processing stage generators
    # src should be hdf reader or other
    # Intermediate stages apply affects
    # Final stage is a driver from drivers.py
    # src is the original source,
    # the source of each stage is the prior stage
    for stage in stages:
        src = stage(src)
    # return the final stage, pull from it to operate the chain
    return src


def apply_processing(hdf_fn):
    with h5py.File(hdf_fn, "r+") as f:
        frames = f["rawimg"]
        vmin, vmax = get_vmin_vmax(f)  # TODO get norm
        total_frames = len(frames)

        bin_size = 15
        pipeline = chain(
            read_h5(frames),
            partial(normalize, vmin=vmin, vmax=vmax),
            partial(spatial_median, k=3),
            partial(frame_binning_avg, bin_size=bin_size),
            to_uint16,
        )

        n = write_h5(pipeline, f, "pipeline-full", total_frames=total_frames // bin_size, compression="lzf")
        print(f"wrote {n} frames")
        ut = f["ut1_unix"][:]
        ut_binned = ut[: n * bin_size].reshape(n, bin_size).mean(axis=1)

        name = "pipeline-full-ut1_unix"
        if name in f:
            del f[name]
        f.create_dataset(name, data=ut_binned)

        f["pipeline-full"].attrs["bin_size"] = bin_size
