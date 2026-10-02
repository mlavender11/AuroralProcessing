"""
===============================================================================
 TV-L1 OPTICAL FLOW  --  a lesson in one runnable file
===============================================================================

Run it top to bottom:

    pip install opencv-contrib-python numpy
    python tvl1_optical_flow.py

Takes about a minute. No GPU. Writes three PNGs next to the script.

INSTALL GOTCHA (this one costs people an hour):
    DualTVL1 lives in the `optflow` module, which ships in opencv_contrib.
    If you have BOTH `opencv-python` and `opencv-contrib-python` installed,
    they overwrite each other's binaries and `cv2.optflow` ends up EMPTY --
    it still imports, it just has nothing in it:

        >>> import cv2; [n for n in dir(cv2.optflow) if 'TVL1' in n]
        []

    Fix: keep exactly one.
        pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
        pip install opencv-contrib-python

    The factory function has also moved between versions; §0 probes for it.

CONTENTS
    §0  environment probe
    §1  the problem: brightness constancy and the aperture problem
    §2  the TV-L1 energy, and why L1 twice
    §3  how it is actually minimised, with the key step in nine lines of numpy
    §4  synthetic data with exact ground truth
    §5  first run -- and the failure mode you will actually hit
    §6  every parameter, mapped back to the math, measured
    §7  where L1 wins, and where it loses badly (measured both ways)
    §8  practical recipes: colour coding, warping, occlusion detection, video

Every number printed by this script was produced by this script. Where TV-L1
loses, it says so.
"""

import os
import sys
import time

import numpy as np
import cv2

OUT = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# §0  ENVIRONMENT PROBE
# =============================================================================
# The factory has lived in four places across OpenCV versions. Try them in
# order instead of hard-coding one and getting an AttributeError.

def make_tvl1(**kwargs):
    """Return a configured DualTVL1 flow object, whichever OpenCV this is."""
    ctor = None
    for path in ("optflow.DualTVL1OpticalFlow_create",   # 4.x / 5.x contrib
                 "optflow.createOptFlow_DualTVL1",       # alias
                 "DualTVL1OpticalFlow_create",           # 3.x, pre-split
                 "createOptFlow_DualTVL1"):              # 3.0-ish
        obj = cv2
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            ctor = obj
            break
        except AttributeError:
            continue
    if ctor is None:
        sys.exit(status="No DualTVL1 in this OpenCV build -- see INSTALL GOTCHA above.")
    flow = ctor()
    # Everything is configured through setters, never constructor args.
    # ('lambda' is a Python keyword, so accept lambda_ as an alias.)
    for k, v in kwargs.items():
        k = "lambda" if k == "lambda_" else k
        getattr(flow, "set" + k[0].upper() + k[1:])(v)
    return flow


print(f"OpenCV {cv2.__version__}")
_p = make_tvl1()
print("TV-L1 defaults:", {
    "tau": _p.getTau(), "lambda": _p.getLambda(), "theta": _p.getTheta(),
    "epsilon": _p.getEpsilon(), "nscales": _p.getScalesNumber(),
    "warps": _p.getWarpingsNumber(), "innerIterations": _p.getInnerIterations(),
    "outerIterations": _p.getOuterIterations(), "scaleStep": _p.getScaleStep(),
    "medianFiltering": _p.getMedianFiltering(), "gamma": _p.getGamma()})


# =============================================================================
# §1  THE PROBLEM
# =============================================================================
#
# Optical flow: for every pixel x = (x, y) in frame I0, find the displacement
# u(x) = (u1, u2) saying where that pixel went in frame I1.
#
# The only assumption available is BRIGHTNESS CONSTANCY -- a point on a surface
# keeps its intensity as it moves:
#
#     I1(x + u(x))  =  I0(x)                                            (1)
#
# One scalar equation per pixel; u has two unknowns per pixel. Underdetermined
# by construction. That is the APERTURE PROBLEM: through a small hole you can
# only measure motion perpendicular to an edge; motion along it is invisible.
#
# Linearising (1) around u = 0 gives the classic optical flow constraint
#
#     I_x*u1 + I_y*u2 + I_t = 0                                         (2)
#
# -- a single line in (u1, u2) space. Every method is a different answer to
# "which point on that line do I pick?":
#
#   Lucas-Kanade  assume u constant on a small window, least-squares all the
#                 constraints in it. Local, sparse, fast; fails on untextured
#                 regions where the window's system is rank-deficient.
#   Horn-Schunck  add a global smoothness penalty ∫|∇u|² and minimise the sum.
#                 Dense -- but quadratic penalties everywhere, so one bad pixel
#                 dominates and motion boundaries smear into ramps.
#   TV-L1         same shape as Horn-Schunck, both terms in L1.
#
# Note (2) is only valid for displacements small enough that the linearisation
# holds -- roughly sub-pixel. Every practical method therefore wraps the solver
# in a coarse-to-fine pyramid. §5 is what happens when that pyramid is too
# shallow for your scene, which is the single most common way this goes wrong.


# =============================================================================
# §2  THE TV-L1 ENERGY, AND WHY L1 TWICE
# =============================================================================
#
# Zach, Pock & Bischof (2007) minimise
#
#     E(u) = ∫ |∇u1| + |∇u2| dx   +   λ ∫ |I1(x + u(x)) − I0(x)| dx     (3)
#            \_________________/       \_______________________/
#              TV regulariser              L1 data term
#
# Two decisions, both load-bearing.
#
# --- Why L1 on the DATA term ---------------------------------------------
# Brightness constancy is not a little wrong everywhere; it is exactly right on
# most of the image and catastrophically wrong on a few pixels -- ones that got
# occluded, specular highlights, impulse noise, things leaving the frame.
#
# A quadratic data term is a MEAN: its influence function grows linearly with
# the residual, so a pixel with residual 50 pulls fifty times harder than one
# with residual 1. A handful of occluded pixels can drag a whole region. An L1
# data term is a MEDIAN: the derivative of |r| is just ±1 no matter how wrong r
# is, so influence is bounded. Gross outliers get outvoted instead of getting a
# megaphone.
#
# Read that sentence carefully, because it contains the limit of the method:
# outvoted BY THE INLIERS. L1 buys robustness to SPARSE violations. It buys you
# nothing when the model is wrong at every pixel at once -- there are no
# inliers left to do the outvoting. §7 measures both cases, and TV-L1 loses the
# second one badly.
#
# --- Why TV (L1 of the gradient) on the SMOOTHNESS term -------------------
# Consider a true motion boundary of height h -- an object sliding across a
# background. Suppose an estimate spreads it over a ramp of width w, so
# |∇u| ≈ h/w over a strip of area ~w:
#
#     quadratic penalty:  (h/w)² · w  =  h²/w   -->  ∞  as w --> 0
#     TV penalty:         (h/w)  · w  =  h      -->  constant
#
# A quadratic regulariser is infinitely opposed to a sharp jump, so it always
# smears it. TV charges the same for a step as for a gradual ramp of the same
# total height: discontinuities are free. That is exactly why TV-L1 fields look
# like piecewise-smooth objects with crisp silhouettes, while Horn-Schunck
# output looks like someone breathed on it.
#
# The price: (3) is convex but non-differentiable in both terms. You cannot
# take a gradient and descend.


# =============================================================================
# §3  HOW IT IS ACTUALLY MINIMISED
# =============================================================================
#
# Three ideas stacked. OpenCV implements all of them, but every parameter in §6
# is a knob on one of them, so they are worth knowing.
#
# --- (a) Linearise the data term, then re-linearise (WARPING) -------------
# Around a current estimate u0, Taylor-expand the warped image:
#
#     I1(x + u) ≈ I1(x + u0) + ∇I1(x + u0)·(u − u0)  =:  ρ(u)           (4)
#
# ρ is affine in u, so |ρ(u)| is convex and piecewise linear. But (4) is only
# good near u0 -- so solve, update u0, re-warp I1, expand again. That outer
# loop is `warps` (default 5).
#
# --- (b) Convex relaxation: split the two hard terms ----------------------
# Introduce a duplicate v of u, coupled by a quadratic:
#
#     E_θ(u,v) = ∫|∇u|  +  (1/2θ)∫(u−v)²  +  λ∫|ρ(v)|                   (5)
#
# As θ --> 0 the coupling forces v --> u and (5) tends to (3). For fixed θ you
# alternate, and each half is something you can solve exactly.
#
# --- (b1) Fix u, solve for v: a POINTWISE SOFT-THRESHOLD ------------------
# min_v (1/2θ)(u−v)² + λ|ρ(v)| decouples across pixels. Closed form, with
# N = |∇I1|²:
#
#     v = u + {  λθ ∇I1                if ρ(u) <  −λθ N
#             { −λθ ∇I1                if ρ(u) >  +λθ N
#             { −ρ(u) ∇I1 / N          otherwise
#
# Geometrically: project u onto the line ρ = 0 -- the optical flow constraint
# line from §1 -- but move AT MOST λθ|∇I1| to get there. That cap is the entire
# robustness story in one clause. A pixel whose data term is screaming (large
# ρ, i.e. an occlusion) may pull the flow by a bounded amount and no further.
#
# --- (b2) Fix v, solve for u: TV DENOISING (the ROF model) ----------------
# min_u ∫|∇u| + (1/2θ)∫(u−v)² is exactly Rudin-Osher-Fatemi denoising of the
# noisy "image" v. Chambolle's 2004 dual algorithm solves it by gradient ascent
# on a dual field p with a projection:
#
#     p ← (p + (τ/θ)∇(div p − v/θ)) / (1 + (τ/θ)|∇(div p − v/θ)|)
#     u  = v − θ div p
#
# which converges provided τ ≤ 1/4 in 2D. Which is where OpenCV's default
# `tau = 0.25` comes from. That is not a tuned magic number, it is the
# stability limit of the dual iteration. Leave it alone.
#
# --- (c) Coarse-to-fine PYRAMID -------------------------------------------
# (4) is only valid for displacements small relative to image structure.
# Downsample both frames by `scaleStep` (0.8), `nscales` times (5); at the
# coarsest level a 40-pixel motion is a 16-pixel motion. Solve, upsample the
# flow (scaling its magnitude), use as the initialisation one level down.
# Largest displacement this can reach, roughly:
#
#     max_disp  ~  (1 / scaleStep)^(nscales − 1)  ×  (a few pixels)
#     defaults:    (1/0.8)^4 ≈ 2.4×  →  under ten pixels
#
# That default reach is SMALL. §5 shows what it looks like when you exceed it.
#
# Here is (b1), the thresholding step, in numpy. You will not call this --
# OpenCV's C++ is hundreds of times faster -- but running it once makes the
# math concrete, and §5 checks it against the real solver's output.

def thresholding_step(u, I0, I1w, I1wx, I1wy, u0, lam, theta):
    """One pointwise soft-threshold, eq. (b1). u, u0 are (H,W,2) float32."""
    rho = (I1w - I0) + I1wx * (u[..., 0] - u0[..., 0]) \
                     + I1wy * (u[..., 1] - u0[..., 1])
    N = I1wx ** 2 + I1wy ** 2 + 1e-12          # |∇I1|², guarded
    lt = lam * theta
    d = np.where(rho < -lt * N,  lt,           # three cases, branchless
        np.where(rho >  lt * N, -lt, -rho / N))
    v = np.empty_like(u)
    v[..., 0] = u[..., 0] + d * I1wx
    v[..., 1] = u[..., 1] + d * I1wy
    return v


# =============================================================================
# §4  SYNTHETIC DATA WITH EXACT GROUND TRUTH
# =============================================================================
# To say anything quantitative we need frames whose true flow we know. This
# scene deliberately contains the two things that break naive methods:
#   * a MOTION DISCONTINUITY -- a rigid object sliding over a background that
#     is itself moving differently
#   * a real OCCLUSION -- background pixels visible in I0 that the object
#     covers in I1, where brightness constancy simply has no answer
#
# Construction trick: to guarantee I0(x) = I1(x + F(x)) EXACTLY, build I0 by
# BACKWARD-warping the source texture by F. (Forward-warping I0 into I1 would
# scatter pixels and leave holes.)

H, W = 288, 384
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def texture(seed):
    """Broadband texture: fine grain + blobs, so the flow is well-conditioned."""
    r = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(r.random((H, W)).astype(np.float32), (0, 0), 1.2)
    coarse = cv2.GaussianBlur(r.random((H, W)).astype(np.float32), (0, 0), 9.0)
    t = 0.55 * fine + 0.45 * coarse
    t -= t.min()
    t /= t.max()
    return (40 + 190 * t).astype(np.float32)


def make_scene(obj_u=(-9.0, 5.0), p0=(150, 120), radius=52):
    """-> I0, I1 (float32 gray), ground-truth flow (H,W,2), occlusion mask."""
    bg = texture(1)
    obj = texture(2) * 0.55 + 110.0            # brighter -> a visible object

    # Background: slow zoom-out about the centre plus a pan. Smooth, small.
    F = np.empty((H, W, 2), np.float32)
    F[..., 0] = 1.6 + 0.012 * (XX - W / 2)
    F[..., 1] = 0.5 + 0.012 * (YY - H / 2)

    # Object: a disc translating by a constant, larger, DIFFERENT displacement.
    obj_u = np.float32(obj_u)
    p1 = (int(p0[0] + obj_u[0]), int(p0[1] + obj_u[1]))
    M0 = np.zeros((H, W), np.uint8); cv2.circle(M0, p0, radius, 255, -1)
    M1 = np.zeros((H, W), np.uint8); cv2.circle(M1, p1, radius, 255, -1)
    M0, M1 = M0 > 0, M1 > 0

    def at(img, p):                             # place texture centred on p
        return np.roll(np.roll(img, p[1] - H // 2, 0), p[0] - W // 2, 1)

    I1 = np.where(M1, at(obj, p1), bg)          # background at rest, obj at p1
    bg0 = cv2.remap(bg, XX + F[..., 0], YY + F[..., 1],
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    I0 = np.where(M0, at(obj, p0), bg0)         # background warped, obj at p0

    gt = F.copy()
    gt[M0] = obj_u                              # object pixels move by obj_u

    # Occlusion: I0 pixels that are background but land inside the object in I1
    tx = np.clip((XX + gt[..., 0]).round().astype(int), 0, W - 1)
    ty = np.clip((YY + gt[..., 1]).round().astype(int), 0, H - 1)
    occ = (~M0) & M1[ty, tx]
    return I0.astype(np.float32), I1.astype(np.float32), gt, occ, M0


I0, I1, GT, OCC, OBJ = make_scene()
g0 = np.clip(I0, 0, 255).astype(np.uint8)
g1 = np.clip(I1, 0, 255).astype(np.uint8)

VALID = ~OCC                                    # brightness constancy holds
BG = (~OBJ) & VALID                             # honest background pixels
# "BAND": background pixels within ~12 px of the object, not occluded. This is
# the CONTAMINATION metric -- innocent bystanders next to the trouble. It is
# the number that actually tests the claims in §2, and it is the one that
# aggregate EPE hides.
BAND = (cv2.dilate(OBJ.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) \
       & (~OBJ) & VALID

print(f"\nScene {W}x{H}: background |u| <= "
      f"{np.linalg.norm(GT[BG], axis=1).max():.1f} px, object |u| = "
      f"{np.linalg.norm(GT[OBJ][0]):.1f} px, "
      f"occluded {OCC.sum()} px ({100*OCC.mean():.1f}%), band {BAND.sum()} px")


# --- error metric ---------------------------------------------------------
# EPE (endpoint error): mean Euclidean distance between estimated and true
# displacement. The standard number. Always report it per region -- a single
# average over the whole frame is 97% easy background and hides everything
# interesting.

def epe(flow, mask=None, gt=GT):
    e = np.linalg.norm(flow - gt, axis=2)
    return float(e.mean() if mask is None else e[mask].mean())


def report(name, flow, ms=None):
    t = f"{ms:6.0f} ms  " if ms is not None else " " * 11
    print(f"  {name:<26}{t}EPE all {epe(flow):7.3f} | obj {epe(flow, OBJ):7.3f}"
          f" | bg {epe(flow, BG):6.3f} | band {epe(flow, BAND):6.3f}"
          f" | occ {epe(flow, OCC):6.3f}")


def run(name, **kw):
    f = make_tvl1(**kw)
    t = time.perf_counter()
    fl = f.calc(g0, g1, None)
    report(name, fl, (time.perf_counter() - t) * 1000)
    return fl


# =============================================================================
# §5  FIRST RUN -- AND THE FAILURE MODE YOU WILL ACTUALLY HIT
# =============================================================================
# The API: construct, configure with setters, call calc(). Input must be 8-bit
# single channel. The convention is  I0(x) ≈ I1(x + flow(x))  -- flow points
# from I0 into I1.

print("\n§5  out of the box")
flow_default = run("TV-L1 defaults")

# Look at the per-region columns, not the aggregate.
#
# The background is essentially PERFECT -- EPE around 0.02 px, a fiftieth of a
# pixel. That is TV-L1 doing what it is famous for.
#
# The object is a disaster: EPE ≈ 8.5 px on a 10.3 px displacement, meaning the
# solver returned roughly ZERO motion there. The disc simply did not move as
# far as TV-L1 is concerned.
#
# This is not a bug and it is not a bad λ. It is the displacement reach from
# §3(c). Defaults give (1/0.8)^4 ≈ 2.4× total downsampling; at the coarsest
# level the disc still moves ~4 px against blurred structure, the linearisation
# in (4) never sees it, and the TV prior happily explains the object as "part
# of the smooth background". Turning λ up does not help -- it makes it worse,
# because a stronger data term just fits the wrong linearisation harder.
#
# The fix is pyramid depth, not weighting. Either drop scaleStep or add scales:

print("\n§5  the fix is pyramid REACH, not lambda")
run("lambda 0.15 -> 0.40", lambda_=0.40)           # no help; slightly worse
flow_step = run("scaleStep 0.8 -> 0.5", scaleStep=0.5)   # works
flow = run("nscales 5 -> 8", scalesNumber=8)             # works better

# Both fixes buy reach, by different routes:
#     scaleStep 0.8 -> 0.5   : (1/0.5)^4 = 16×   -- four aggressive levels
#     nscales   5   -> 8     : (1/0.8)^7 ≈ 4.8×  -- seven gentle levels
#
# Object EPE drops from ~8.5 px to ~0.25 px either way. But compare the `band`
# column: the gentle deep pyramid keeps detail near the motion boundary roughly
# three times better than the aggressive shallow one, because each level throws
# away less between steps and the flow has more chances to be corrected on the
# way down. That generalises:
#
#     MORE LEVELS AT A GENTLE RATIO BEATS FEWER LEVELS AT AN AGGRESSIVE ONE.
#     Reach for `nscales` before you reach for `scaleStep`.
#
# The underlying trade-off, which is the thing to remember:
#
#     TOO SHALLOW  -> large motions silently missed; the output looks smooth
#                     and plausible while being completely wrong
#     TOO DEEP     -> fine detail smoothed away, small structures lost
#
# Failing SILENTLY is what makes the first one dangerous. A too-shallow pyramid
# does not error out; it hands you a beautiful, smooth, confident, wrong flow
# field. ALWAYS estimate your largest expected displacement and set the pyramid
# for it. `flow` (the fixed one) is what the rest of this file uses.

BASE = dict(scalesNumber=8)     # our working configuration for this scene

# --- check the numpy thresholding step from §3 ---------------------------
# It should drive the LINEARISED residual ρ toward zero, except where the
# λθ|∇I1| cap binds -- which is precisely the robustness mechanism.
I1w = cv2.remap(I1, XX + flow[..., 0], YY + flow[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
I1wx = cv2.Sobel(I1w, cv2.CV_32F, 1, 0, ksize=3) / 8.0
I1wy = cv2.Sobel(I1w, cv2.CV_32F, 0, 1, ksize=3) / 8.0
lam, theta = 0.15, 0.3
v = thresholding_step(flow, I0, I1w, I1wx, I1wy, flow, lam, theta)
rho0 = (I1w - I0)
rho1 = rho0 + I1wx * (v[..., 0] - flow[..., 0]) + I1wy * (v[..., 1] - flow[..., 1])
capped = np.abs(rho0) > lam * theta * (I1wx ** 2 + I1wy ** 2)
print(f"\n§5  numpy thresholding step: mean |rho| {np.abs(rho0).mean():.2f} -> "
      f"{np.abs(rho1).mean():.2f} gray levels; the lambda*theta cap binds on "
      f"{100*capped.mean():.0f}% of pixels,")
print(f"    and no pixel is allowed to move more than "
      f"{lam*theta*np.hypot(I1wx, I1wy).max():.2f} px in one step. "
      f"Bounded influence -- that IS the L1 data term.")


# =============================================================================
# §6  EVERY PARAMETER, MAPPED BACK TO THE MATH
# =============================================================================
#
#  setter              default  what it is in §3                  advice
#  ------------------  -------  --------------------------------  ------------
#  setTau              0.25     step of Chambolle's dual ascent   don't touch;
#                               (b2); stability needs τ ≤ 1/4     it's a proof
#  setLambda           0.15     λ in (3): data vs smoothness.     the one knob
#                               Larger = trust pixels more =      worth tuning
#                               more detail AND more noise
#  setTheta            0.30     coupling tightness in (5). Small  rarely touch
#                               = closer to the true energy but
#                               slower and less stable
#  setEpsilon          0.01     stopping threshold on the update  0.005 floor
#  setScalesNumber     5        pyramid levels (c)                SET THESE
#  setScaleStep        0.8      downsample ratio per level        FOR YOUR
#  setWarpingsNumber   5        re-linearisations per level (a)   MOTION (§5)
#  setInnerIterations  30       (b1)+(b2) alternations
#  setOuterIterations  10       outer loop, checked against ε
#  setMedianFiltering  5        k×k median on the flow each warp; keep it;
#                               k ∈ {1,3,5} only, 7+ throws. Not  1 disables
#                               in the paper -- a heuristic from
#                               Sun/Roth/Black "Secrets of
#                               optical flow" that kills outliers
#                               for almost nothing
#  setGamma            0.0      >0 adds an illumination-variation see §7 --
#                               term                              it disappoints
#  setUseInitialFlow   False    True = seed calc() with the flow  use it for
#                               array you pass in                 video (§8d)
#
# Measured on this scene, from the BASE config:

print("\n§6  lambda -- the data/smoothness trade-off")
for x in (0.03, 0.08, 0.15, 0.30, 0.60):
    run(f"lambda={x}", **BASE, lambda_=x)

print("\n§6  pyramid reach: nscales at the default gentle ratio")
for x in (3, 5, 8, 11):
    run(f"nscales={x}", scalesNumber=x)

print("\n§6  pyramid reach: scaleStep at the default 5 levels")
for x in (0.9, 0.8, 0.65, 0.5):
    run(f"scaleStep={x}", scaleStep=x)

print("\n§6  warps -- re-linearisations per level")
for x in (1, 3, 5, 9):
    run(f"warps={x}", **BASE, warpingsNumber=x)

print("\n§6  epsilon -- accuracy vs time")
for x in (0.05, 0.01, 0.005, 0.001):
    run(f"epsilon={x}", **BASE, epsilon=x)

print("\n§6  median filtering (the 'secret' heuristic)")
for x in (1, 3, 5):
    run(f"medianFiltering={x}", **BASE, medianFiltering=x)

# Read those tables rather than skimming them:
#   * lambda is U-shaped, not monotonic. Too small and the regulariser wins and
#     everything flattens; too large and you fit noise -- and it gets much
#     slower, because the data term keeps yanking the solution around. The
#     optimum is scene-dependent. Sweep it on YOUR data.
#   * epsilon past ~0.005 buys nothing but time -- 0.001 costs roughly 10× and
#     lands on the same answer. Convergence tolerance is not what limits your
#     accuracy; linearisation and pyramid depth are.
#   * medianFiltering barely moves the needle on clean synthetic data -- here
#     disabling it can even score marginally better. It earns its keep on real,
#     noisy footage, which is exactly why it is a heuristic bolted onto the
#     algorithm rather than a term in the energy. Leave it on for real video.


# =============================================================================
# §7  WHERE L1 WINS, AND WHERE IT LOSES BADLY
# =============================================================================
# Compared against Farneback (quadratic polynomial fitting + quadratic
# smoothing -- a fair stand-in for "the L2 way") and DIS (the fast modern
# default in core OpenCV).

def farneback(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 5, 21, 5, 7, 1.5, 0)


def dis(a, b):
    return cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)


print("\n§7  test 1: motion discontinuity + occlusion (the scene as built)")
fb = farneback(g0, g1)
fd = dis(g0, g1)
report("TV-L1", flow)
report("Farneback", fb)
report("DIS medium", fd)
# The column that tests §2 is BAND -- non-occluded background right next to the
# occlusion. TV-L1 should keep the damage local while the quadratic method
# lets it bleed outward. It does, by a comfortable margin.
#
# Note the honest wrinkle in the `occ` column: Farneback often scores BETTER on
# the occluded pixels themselves. That is not skill. Those pixels are invisible
# in I1, so no method can know their motion; ground truth happens to be the
# smooth background flow, and heavy Gaussian smoothing happens to interpolate
# exactly that. It is a metric artefact. Judge occlusion handling by the
# contamination of the pixels around it, which is what BAND measures.

print("\n§7  test 2: SPARSE gross outliers (2% salt-and-pepper on I1)")
# This is the case L1 was designed for: the model is right almost everywhere
# and violently wrong at a few isolated pixels.
r = np.random.default_rng(7)
n = r.random((H, W))
g1n = g1.copy()
g1n[n < 0.01] = 0
g1n[n > 0.99] = 255
report("TV-L1", make_tvl1(**BASE).calc(g0, g1n, None))
report("Farneback", farneback(g0, g1n))
# Compare each row against its own clean result in test 1. TV-L1 absorbs less
# absolute damage, and -- the headline -- TV-L1 with 2% of its pixels destroyed
# is still more accurate than Farneback was on the untouched pair. Bounded
# influence, doing exactly what §2 promised.

print("\n§7  test 3: GLOBAL illumination change (I1 -> 1.25*I1 + 12)")
# And here is the case people wrongly assume L1 also covers. Brightness
# constancy is now violated at EVERY pixel. There are no inliers to outvote the
# outliers, so robustness has nothing to work with.
g1_lit = np.clip(1.25 * I1 + 12, 0, 255).astype(np.uint8)
report("Farneback", farneback(g0, g1_lit))
report("DIS medium", dis(g0, g1_lit))
for gm in (0.0, 0.02, 0.5):
    report(f"TV-L1 gamma={gm}", make_tvl1(**BASE, gamma=gm).calc(g0, g1_lit, None))
# TV-L1 does not degrade here -- it DIVERGES. The reported EPE is larger than
# the largest true displacement in the scene, i.e. the output is worse than
# returning all zeros. Deeper pyramids make it worse still (coarse levels turn
# the constant offset into apparent motion, and every finer level inherits it;
# try scaleStep=0.5 here and watch the EPE quadruple). Farneback, whose
# polynomial coefficients partly cancel a gain, barely notices. So under
# lighting change TV-L1 is the most brittle of the three, not the most robust.
# `gamma`, the gradient-constancy term, helps only partially and costs time.
#
# The actual fix is preprocessing -- remove the low frequencies, where
# illumination lives, before you ever call calc():

print("\n§7  test 3, fixed: high-pass both frames first")


def highpass(img, sigma=8.0):
    """Subtract the local mean. Kills gain/offset; keeps the texture flow needs."""
    f = img.astype(np.float32)
    return np.clip(128 + 2.0 * (f - cv2.GaussianBlur(f, (0, 0), sigma)),
                   0, 255).astype(np.uint8)


report("TV-L1 + highpass", make_tvl1(**BASE).calc(highpass(g0), highpass(g1_lit), None))
report("Farneback + highpass", farneback(highpass(g0), highpass(g1_lit)))
# Back to normal, and slightly better than the untouched pair. On real footage
# use this, or the structure-texture decomposition from the original paper, or
# CLAHE. Preprocessing is not an optional polish step for TV-L1; it is part of
# the method.
#
# THE TAKEAWAY FROM §7, stated precisely:
#   L1 buys robustness to SPARSE violations of brightness constancy.
#   It buys nothing against DENSE ones. Fix those in preprocessing.


# =============================================================================
# §8  PRACTICAL RECIPES
# =============================================================================

# --- 8a. Middlebury colour coding ----------------------------------------
# Hue = direction, saturation = magnitude. Do NOT roll your own linear HSV map:
# the Baker et al. colour wheel is deliberately non-uniform so the cardinal
# directions stay perceptually distinguishable. Here it is, properly.

def _color_wheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    w = np.zeros((RY + YG + GC + CB + BM + MR, 3), np.float32)   # R, G, B
    c = 0
    t = np.arange(RY)/RY; w[c:c+RY, 0] = 255; w[c:c+RY, 1] = 255*t;     c += RY
    t = np.arange(YG)/YG; w[c:c+YG, 0] = 255*(1-t); w[c:c+YG, 1] = 255; c += YG
    t = np.arange(GC)/GC; w[c:c+GC, 1] = 255; w[c:c+GC, 2] = 255*t;     c += GC
    t = np.arange(CB)/CB; w[c:c+CB, 1] = 255*(1-t); w[c:c+CB, 2] = 255; c += CB
    t = np.arange(BM)/BM; w[c:c+BM, 2] = 255; w[c:c+BM, 0] = 255*t;     c += BM
    t = np.arange(MR)/MR; w[c:c+MR, 2] = 255*(1-t); w[c:c+MR, 0] = 255
    return w


_WHEEL = _color_wheel()


def flow_to_color(flow, max_mag=None):
    """Middlebury colour coding -> BGR uint8. Pass max_mag to compare fields."""
    u, v = flow[..., 0], flow[..., 1]
    if max_mag is None:
        max_mag = max(np.sqrt(u**2 + v**2).max(), 1e-6)
    u, v = u / max_mag, v / max_mag
    rad = np.sqrt(u**2 + v**2)[..., None]
    a = np.arctan2(-v, -u) / np.pi                         # [-1, 1]
    nc = _WHEEL.shape[0]
    fk = (a + 1) / 2 * (nc - 1)
    k0 = np.floor(fk).astype(int)
    f = (fk - k0)[..., None]
    col = (1 - f) * _WHEEL[k0] + f * _WHEEL[(k0 + 1) % nc]  # (H,W,3) RGB
    col = np.where(rad <= 1, 255 - rad * (255 - col), col * 0.75)
    return np.clip(col[..., ::-1], 0, 255).astype(np.uint8)  # -> BGR


# --- 8b. Warping a frame with the flow -----------------------------------
# Your visual correctness check: if the flow is right, warping I1 by it must
# reproduce I0. Note the sign, and that remap wants ABSOLUTE coordinates, not
# offsets. That is the single most common bug in optical flow code.

def warp(img, flow):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + flow[..., 0], yy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


resid = np.abs(warp(g1, flow).astype(np.float32) - g0.astype(np.float32))
print(f"\n§8  warp check: mean |I1(x+u) − I0(x)| = {resid.mean():.2f} gray levels"
      f"  (non-occluded only: {resid[VALID].mean():.2f})")
# On real footage you have no ground truth, so this residual is your only
# quantitative handle. Track it. A sudden jump means the flow broke.

# --- 8c. Forward-backward consistency = free occlusion detection ----------
# Compute flow both ways. For a non-occluded pixel, going forward then backward
# returns you where you started. Where it does not, you are looking at an
# occlusion or an outright failure -- and you should distrust that pixel in
# whatever you do downstream (interpolation, tracking, stabilisation).

def fb_consistency(a, b, thresh=1.5, **kw):
    fwd = make_tvl1(**kw).calc(a, b, None)
    bwd = make_tvl1(**kw).calc(b, a, None)
    err = np.linalg.norm(fwd + warp(bwd, fwd), axis=2)
    return fwd, err, err > thresh


fwd, fberr, bad = fb_consistency(g0, g1, **BASE)
print(f"§8  FB-consistency flags {100*bad.mean():.1f}% of pixels: catches "
      f"{100*(bad & OCC).sum()/max(OCC.sum(),1):.0f}% of true occlusions, "
      f"false-alarms on {100*(bad & VALID).sum()/max(VALID.sum(),1):.1f}% of "
      f"good ones -- with zero ground truth required.")

# --- 8d. Video: warm-start every frame -----------------------------------
# Motion is temporally coherent, so the previous frame's flow is an excellent
# initialisation. setUseInitialFlow(True) usually cuts time substantially and
# stabilises the result:
#
#     tv = make_tvl1(scaleStep=0.5, useInitialFlow=True)
#     prev, fl = first_gray_frame, None
#     while ok:
#         ok, frame = cap.read()
#         gray = highpass(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))   # see §7!
#         fl = tv.calc(prev, gray, fl)          # pass the previous flow back in
#         cv2.imshow("flow", flow_to_color(fl))
#         prev = gray
#
# --- 8e. Speed -----------------------------------------------------------
# TV-L1 is SLOW: hundreds of ms per frame on CPU at modest resolution. Two
# things that actually help:
#   * Downscale before, upscale after: run at half size and multiply the flow
#     by 2. It also doubles your effective displacement reach (§5), so on
#     large-motion footage it can be a straight win. It is NOT free, though --
#     on this scene, which is all fine texture, it roughly doubles the EPE for
#     a ~2.5× speedup. Measure it on your own data rather than assuming.
#   * If you need real time, cv2.DISOpticalFlow_create(...) is 1-2 orders of
#     magnitude faster at a modest accuracy cost, and lives in core OpenCV
#     with no contrib dependency. Measured on this scene:

t = time.perf_counter(); make_tvl1(**BASE).calc(g0, g1, None)
t_tv = (time.perf_counter() - t) * 1000
t = time.perf_counter(); dis(g0, g1)
t_dis = (time.perf_counter() - t) * 1000
half0 = cv2.pyrDown(g0); half1 = cv2.pyrDown(g1)
t = time.perf_counter(); fh = make_tvl1(**BASE).calc(half0, half1, None)
t_half = (time.perf_counter() - t) * 1000
fh = cv2.resize(fh, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0
print(f"§8  timing: TV-L1 {t_tv:.0f} ms | TV-L1 at half size {t_half:.0f} ms "
      f"(EPE {epe(fh):.3f} vs {epe(flow):.3f}) | DIS {t_dis:.0f} ms")

# --- 8f. Figures ---------------------------------------------------------
# Shared normalisation so the panels are comparable. Note the 1.1 headroom:
# flow_to_color darkens anything beyond max_mag, and an estimate that overshoots
# ground truth by a hair otherwise speckles the object with dark pixels.
scale = 1.1 * np.linalg.norm(GT, axis=2).max()


def tile(name, img):
    im = img.copy()
    cv2.rectangle(im, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(im, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return im


grid = np.vstack([
    np.hstack([tile("frame I0", cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)),
               tile("frame I1", cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)),
               tile("ground truth", flow_to_color(GT, scale))]),
    np.hstack([tile("TV-L1 (nscales=8)", flow_to_color(flow, scale)),
               tile("Farneback", flow_to_color(fb, scale)),
               tile("fwd-bwd inconsistency", cv2.applyColorMap(
                   np.clip(fberr * 60, 0, 255).astype(np.uint8),
                   cv2.COLORMAP_INFERNO))])])
cv2.imwrite(os.path.join(OUT, "tvl1_overview.png"), grid)

cv2.imwrite(os.path.join(OUT, "tvl1_pyramid.png"), np.hstack([
    tile("defaults: object lost", flow_to_color(flow_default, scale)),
    tile("scaleStep 0.5 (16x reach)", flow_to_color(flow_step, scale)),
    tile("nscales 8 (4.8x reach)", flow_to_color(flow, scale)),
    tile("ground truth", flow_to_color(GT, scale))]))

cv2.imwrite(os.path.join(OUT, "tvl1_lambda.png"), np.hstack([
    tile(f"lambda = {x}", flow_to_color(
        make_tvl1(**BASE, lambda_=x).calc(g0, g1, None), scale))
    for x in (0.03, 0.15, 0.60)]))

print(f"\nWrote tvl1_overview.png, tvl1_pyramid.png, tvl1_lambda.png to {OUT}")

# =============================================================================
# READING
#   Zach, Pock, Bischof (2007)   A duality based approach for realtime TV-L1
#                                optical flow.  -- the algorithm itself
#   Chambolle (2004)             An algorithm for total variation minimization
#                                and applications.  -- the (b2) dual step
#   Sánchez, Meinhardt-Llopis, Facciolo (2013)  TV-L1 Optical Flow Estimation,
#                                IPOL.  -- a line-by-line reference
#                                implementation with a live online demo; the
#                                best companion to this file if you now want to
#                                write the solver yourself
#   Sun, Roth, Black (2010)      Secrets of optical flow estimation and their
#                                principles.  -- where medianFiltering comes
#                                from, and why heuristics beat energies
#   Baker et al. (2011)          A database and evaluation methodology for
#                                optical flow.  -- Middlebury: EPE, the colour
#                                wheel, and real ground-truth sequences to try
#                                this on next
# =============================================================================
