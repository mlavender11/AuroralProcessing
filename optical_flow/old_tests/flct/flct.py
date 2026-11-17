import matplotlib.pyplot as plt
import numpy as np
import pyflct
import cv2
from pathlib import Path
from scipy.ndimage import uniform_filter, median_filter
from itertools import pairwise
from tqdm.auto import tqdm


def extract_frame(vidfn, frame_i, outfn):
    """
    extract frame from video file based on frame index
    """
    cap = cv2.VideoCapture(vidfn)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(outfn, frame)
    else:
        print(f"Failed to extract frame {outfn}")

    # Release the video file template resources
    cap.release()

    return outfn


def read_frame(fn, crop=490):
    """
    reads and crops frame
    """
    f = cv2.imread(fn)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f = f[0:crop, :]
    return f


def get_vels(fn1, fn2, blur_size=9, blur_sigma=0, **kwargs):
    """
    get velocity vectors from flct correlation method
    applys gaussian blur to frames
    returns nd array of [vm, vel_x, vel_y] where vm is the mask, and vel_x/vel_y are the x/y components
    """

    # read in imgs, cvt to gray, crop
    f1 = cv2.imread(fn1)
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f1 = f1[0:490, :]

    f2 = cv2.imread(fn2)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    f2 = f2[0:490, :]

    f1 = cv2.GaussianBlur(f1, (blur_size, blur_size), blur_sigma)
    f2 = cv2.GaussianBlur(f2, (blur_size, blur_size), blur_sigma)

    # optical flow
    vel_x, vel_y, vm = pyflct.flct(f1, f2, **kwargs)
    vels = np.stack([vm, vel_x, vel_y])
    return vels


def display(fn1, fn2, vels, scale=2):
    """
    displays triptych of f1, f1 overlayed with vectors, f2
    """

    f1 = read_frame(fn1)
    f2 = read_frame(fn2)

    height, width = f1.shape

    X = np.arange(0, width, 1)
    Y = np.arange(0, height, 1)
    XX, YY = np.meshgrid(X, Y)

    vm = vels[0]
    vel_x = vels[1]
    vel_y = vels[2]

    vel_x_masked = np.where(vm == 1, vel_x, np.nan)
    vel_y_masked = np.where(vm == 1, vel_y, np.nan)

    vel_x_interpolated = np.where(vm == 0.5, vel_x, np.nan)
    vel_y_interpolated = np.where(vm == 0.5, vel_y, np.nan)

    step = 10

    XX_sub = XX[::step, ::step]
    YY_sub = YY[::step, ::step]
    vel_x_masked_sub = vel_x_masked[::step, ::step]
    vel_y_masked_sub = vel_y_masked[::step, ::step]
    vel_x_interpolated_sub = vel_x_interpolated[::step, ::step]
    vel_y_interpolated_sub = vel_y_interpolated[::step, ::step]

    # 6. Plotting
    fig = plt.figure(figsize=(30, 10))

    # Plotting the first image
    ax1 = fig.add_subplot(131)
    ax1.imshow(f1, cmap="gray", origin="lower")
    ax1.set_title("Frame 1")

    # Plot vectors over the background frame to check structural alignment
    ax2 = fig.add_subplot(132)
    ax2.imshow(f1, cmap="gray", origin="lower", alpha=0.6)  # Added background for context
    ax2.quiver(
        XX_sub, YY_sub, vel_x_masked_sub, vel_y_masked_sub, scale=scale, color="red"
    )  # Changed color for visibility
    ax2.quiver(
        XX_sub, YY_sub, vel_x_interpolated_sub, vel_y_interpolated_sub, scale=scale, color="blue"
    )  # Changed color for visibility
    ax2.set_title("Flow Field Overlay")

    # Plot the shifted image
    ax3 = fig.add_subplot(133)
    ax3.imshow(f2, cmap="gray", origin="lower")
    ax3.set_title("Frame 2")

    plt.show()


def save_output_frame(out_dir, fn1, fn2, i1, i2, vels, scale=10):
    print(f"saving frame {fn1}")
    # TODO add frame index to title
    f1 = read_frame(fn1)
    f2 = read_frame(fn2)
    height, width = f1.shape

    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)

    vm, vel_x, vel_y = vels

    vel_x_masked = np.where(vm == 1, vel_x, np.nan)
    vel_y_masked = np.where(vm == 1, vel_y, np.nan)

    vel_x_interpolated = np.where(vm == 0.5, vel_x, np.nan)
    vel_y_interpolated = np.where(vm == 0.5, vel_y, np.nan)

    step = 10  # vector downsampling
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    vel_x_masked_sub = vel_x_masked[::step, ::step]
    vel_y_masked_sub = vel_y_masked[::step, ::step]
    vel_x_interpolated_sub = vel_x_interpolated[::step, ::step]
    vel_y_interpolated_sub = vel_y_interpolated[::step, ::step]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(f1, cmap="gray", origin="lower")  # add alpha?
    ax.quiver(X_sub, Y_sub, vel_x_masked_sub, vel_y_masked_sub, scale=scale, color="blue")
    ax.set_title(f"frame {i1} -> frame {i2}")
    fig.savefig(out_dir / f"frame{i1}-{i2}.png")
    plt.close(fig)


def make_frames(frame_dir, out_dir, start_frame, end_frame, frame_step, **kwargs):
    frame_pairs = list(pairwise(range(start_frame, end_frame + 1, frame_step)))
    for f1_index, f2_index in frame_pairs:
        fn1 = frame_dir / f"frame{f1_index}.png"
        fn2 = frame_dir / f"frame{f2_index}.png"

        vels = get_vels(fn1, fn2, **kwargs)
        save_output_frame(out_dir, fn1, fn2, f1_index, f2_index, vels)


def main():
    print("running main")
    frame_dir = Path("/Users/michaellavender/Documents/BUSPC/optical flow/flct/frames")
    start_frame = 3195
    end_frame = 3313
    frame_step = 8
    deltat = frame_step * 1  # 1s per frame
    out_dir = Path(f"/Users/michaellavender/Documents/BUSPC/optical flow/flct/output_frames-stepsize{frame_step}")
    out_dir.mkdir(parents=True, exist_ok=True)

    make_frames(
        frame_dir,
        out_dir,
        start_frame,
        end_frame,
        frame_step,
        deltat=deltat,
        deltas=1,
        sigma=8,
        thresh=0.7,
        biascor=False,
        interp=False,
    )


if __name__ == "__main__":
    main()
