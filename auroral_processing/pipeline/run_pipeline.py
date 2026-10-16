from auroral_processing.pipeline.processing_chain import apply_processing
from auroral_processing.products.video import make_video_from_source
from auroral_processing.io.hdf_utilities import repack

# fn = "/Users/michaellavender/Developer/AuroralProcessing/optical_flow/data/2013-03-27-CamSer7196_015400-015600.h5"
fn = "/Users/michaellavender/Developer/AuroralProcessing/optical_flow/data/2013-03-27-CamSer7196_015400-015600-copy.h5"

iteration = "v4"


if __name__ == "__main__":

    chain = [
        {"stage": "normalize"},
        {"stage": "spatial_median", "k": 3},
        {"stage": "binned_temporal_average", "bin_size": 30},
        {"stage": "wiener"},
        {"stage": "to_uint16"},
    ]




    print("proc")
    apply_processing(fn, chain, iteration)
    print("repack")
    repack(fn, replace=True)  # reclaims space from deleted/overwritten datasets
    print("video")
    make_video_from_source(fn, out_fn=f"{iteration}.mp4", img_dset=iteration, ut_dset=f"{iteration}_ut1_unix")
    print("optical flow:")
    print(f"  python opencv-opt-flow.py {fn} --images /{iteration} " f"--time /{iteration}_ut1_unix --unix-time UT1")


# TODO check write atts, fit stages
