import cv2
import h5py
import numpy as np
from tqdm.auto import tqdm


def make_tvl1(**kwargs):
    flow = cv2.optflow.createOptFlow_DualTVL1()
    for k, v in kwargs.items():
        k = "lambda" if k == "lambda_" else k
        getattr(flow, "set" + k[0].upper() + k[1:])(v)

    return flow


def get_defaults():
    flow = make_tvl1()
    print(
        "TV-L1 defaults:",
        {
            "tau": flow.getTau(),
            "lambda": flow.getLambda(),
            "theta": flow.getTheta(),
            "epsilon": flow.getEpsilon(),
            "nscales": flow.getScalesNumber(),
            "warps": flow.getWarpingsNumber(),
            "innerIterations": flow.getInnerIterations(),
            "outerIterations": flow.getOuterIterations(),
            "scaleStep": flow.getScaleStep(),
            "medianFiltering": flow.getMedianFiltering(),
            "gamma": flow.getGamma(),
        },
    )


def read_hdf(hdf_obj, dataset_name):
    yield ...


def get_two_frame_flow(flow_obj, frame1, frame2, prev_flow):
    # need frame 1 and 2 to be uint8
    img0 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)  # How to pass an existing ndarray
    img1 = ...

    flow = flow_obj.calc(img0, img1, prev_flow)

    return flow


def get_sequential_flow(src, flow_obj, use_initial_flow=False):
    flow_list = []
    flow = None
    frame0 = next(src)
    flow_obj.setUseInitialFlow(False)
    for frame1 in src:
        flow = get_two_frame_flow(flow_obj, frame0, frame1, flow)
        flow_list.append(flow.copy())  # or yield flow copy?
        flow_obj.setUseInitialFlow(use_initial_flow)
        frame0 = frame1

    return flow_list


def downsample_flow_field(flow, n):
    """Average the dense field onto an n-ish grid -> X, Y, u, v (like a PIV field)."""
    h, w = flow.shape[:2]
    step = max(h // n, w // n, 1)
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    half = step // 2
    u = np.array(
        [[flow[max(y - half, 0) : y + half + 1, max(x - half, 0) : x + half + 1, 0].mean() for x in xs] for y in ys]
    )
    v = np.array(
        [[flow[max(y - half, 0) : y + half + 1, max(x - half, 0) : x + half + 1, 1].mean() for x in xs] for y in ys]
    )
    X, Y = np.meshgrid(xs, ys)
    return X, Y, u, v


# Lower lambda for noisy footage? Play around with scale sizes
def write_flow_to_hdf(src, hdf_obj: h5py.File, num_frames, flow_params: dict):
    # Are flow objs fp32?
    dataset_name = ...  #

    dset = None
    i = 0
    for flow in tqdm(src, total=num_frames):
        if dset is None:
            frame_shape = flow.shape[:2]
            hdf_obj.create_dataset(
                dataset_name,
                shape=(num_frames - 1, *frame_shape, 2),  # Does this unpacking work
                dtype=np.float32,
                chunks=(1, *frame_shape, 2),
                compression="gzip",  # TODO what's best here
                compression_opts=4,
            )
            dset = hdf_obj[dataset_name]
            dset.attrs.update(flow_params)

        dset[i] = flow
        i += 1

    return i


def make_overlay_video():
    ...