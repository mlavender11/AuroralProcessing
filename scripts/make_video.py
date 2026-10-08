# Command line script for making videos from hist HDF5 files

import argparse
from auroral_processing.products.video import make_video_from_source

parser = argparse.ArgumentParser(description="Make video from HiST HDF5 file")
parser.add_argument("filename", type=str, help="path to HDF5 file")
parser.add_argument("-o", "--out_path", type=str, help="output path of video. Defaults to input with .mp4 suffix")

args = parser.parse_args()

make_video_from_source(args.filename, args.out_path)
