def add_parser(sub):
    p = sub.add_parser("pipeline", help="Run pre-processing pipeline", description="TODO")
    # TODO


def run(args):
    from auroral_processing.pipeline.processing_chain import apply_processing

    # TODO


# parser = argparse.ArgumentParser(description="Pre-Processing frames in HDF5 file")
#     parser.add_argument("hdf_fn", help="path to the HDF5 file")
#     parser.add_argument("-r", "--repack", action="store_true", help="repack the HDF file")
#     parser.add_argument(
#         "-i",
#         "--inplace",
#         action="store_true",
#         help="with --repack: overwrite the original file instead of writing seperate .repacked.h5",
#     )
#     args = parser.parse_args()

#     apply_processing(args.hdf_fn)
#     if args.repack:
#         print("repacking")
#         repack(args.hdf_fn, replace=args.inplace)
