from pathlib import Path
from pprint import pprint

import numpy as np

import histutils.dio
import histutils.index
import histutils.timedmc as hstt
import histutils.rawDMCreader
from histutils.hstxmlparse import xmlparam


def sibling_paths(raw_path):
    dmc = Path(raw_path).expanduser()
    xml  = dmc.with_suffix(".xml").expanduser()
    nmea = dmc.with_suffix(".nmea").expanduser()
    for p in (dmc, xml, nmea):
        if not p.exists():
            raise FileNotFoundError(f"expected companion file missing: {p}")
    return dmc, xml, nmea


def whole_binary_to_hdf(dmc_path, out_dir):
    # path to the data. This will probably be distinct for your computer.
    dmc_fn, xml_fn, nmea_fn = sibling_paths(dmc_path) 

    # where to store the converted data
    out_fn = Path(out_dir) / (Path(dmc_path).stem + ".h5")
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    x = xmlparam(xml_fn)

    # only 2011-era files have 0 header_bytes. Newer have 4 header_bytes.

    params = {
        "header_bytes": 4,  # only 2011-era files have 0 header_bytes. Newer have 4 header_bytes.
        "xy_pixel": (x["horizpixels"], x["vertpixels"]),
        "xy_bin": (x["binning"], x["binning"]),
        "kineticsec": x["kineticrate"],
        "rotccw": 0,  # counter clockwise rotation in degrees
        "transpose": False,
        "flipud": False,  # flip up down
        "fliplr": False,  # flip left right
    }

    gpsInfo = hstt.parse_gprmc(nmea_fn)
    params["startUTC"] = gpsInfo

    fInfo = histutils.rawDMCreader.getDMCparam(dmc_fn, params)
    params.update(fInfo)

    pprint(params)

    if out_fn.is_file():
        raise FileExistsError(
            f"{out_fn} already exists. Please delete or move it before running this script."
        )

    i0, iend = histutils.index.getRawInd(dmc_fn, params)
    print(f"first raw frame index: {i0}, last raw frame index: {iend}")
    iraw = np.arange(i0, iend + 1)

    tUTC = histutils.dio.frame2ut1(params["startUTC"], params["kineticsec"], iraw)
    print(f"raw frames cover {tUTC[0]} to {tUTC[-1]}")


    histutils.dio.vid2h5(dmc_fn, out_fn, rawind=iraw, params=params)

def main():
    dmc_fn = '/Users/michaellavender/Documents/BUSPC/binary2hdfTest/2013-04-11T07-00-CamSer1387_frames_402209-1-403708.DMCdata'
    out_dir = '/Users/michaellavender/Documents/BUSPC/binary2hdfTest'
    print('saving')
    whole_binary_to_hdf(dmc_fn, out_dir)

if __name__ == "__main__":
    main()