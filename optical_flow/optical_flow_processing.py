import h5py
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from functools import partial
from itertools import islice
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import subprocess
import argparse

from auroral_processing.hdf_utils import compute_norm, compute_norm_from_hdf
