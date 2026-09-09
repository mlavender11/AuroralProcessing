#!/usr/bin/env python
# coding: utf-8

# In[3]:


import datetime
import pyaurorax
from pathlib import Path

# # Ft Yukon Keograms

# In[17]:


def make_fykn_keogram(date_str, out_dir):
    try:
        print(f"making fykn keogram for {date_str}")
        aurorax = pyaurorax.PyAuroraX(download_output_root_path="/Volumes/Elements/THEMIS-DATA/Fort-Yukon")
        at = aurorax.tools
        at.set_theme("dark")

        d_object = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        start_time = datetime.time(7, 0, 0)
        end_time = datetime.time(13, 0, 0)
        start_dt = datetime.datetime.combine(d_object, start_time)
        end_dt = datetime.datetime.combine(d_object, end_time)
        out_fn = out_dir / ("FYKN-" + date_str + ".png")

        r = aurorax.data.ucalgary.download("THEMIS_ASI_RAW", start_dt, end_dt, site_uid="fykn")
        data = aurorax.data.ucalgary.read(r.dataset, r.filenames, n_parallel=5)
        images_scaled = at.scale_intensity(data.data, min=10, max=10000)
        keogram = at.keogram.create(images_scaled, data.timestamp)

        duration_hours = (end_dt - start_dt).total_seconds() / 3600
        inches_per_hour = 2  # tune to taste
        n_frames = keogram.data.shape[1]
        desired_ticks = 12
        xtick_increment = max(1, n_frames // desired_ticks)

        keogram.plot(
            title="THEMIS ASI Fort Yukon %s" % (start_dt.strftime("%Y-%m-%d UT%H")),
            figsize=(duration_hours * inches_per_hour, 4),
            aspect="auto",
            cmap="gray",
            savefig_filename=out_fn,
            savefig=True,
            xtick_increment=xtick_increment,
        )
    except Exception as e:
        print(f"Error when making fykn keogram for {date_str}: {e}")


# # Ft Yukon Videos

# In[18]:


def make_fykn_video(date_str, out_dir):
    try:
        print(f"making fykn video for {date_str}")
        aurorax = pyaurorax.PyAuroraX(download_output_root_path="/Volumes/Elements/THEMIS-DATA/Fort-Yukon")
        at = aurorax.tools

        d_object = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        start_time = datetime.time(7, 0, 0)
        end_time = datetime.time(13, 0, 0)
        start_dt = datetime.datetime.combine(d_object, start_time)
        end_dt = datetime.datetime.combine(d_object, end_time)
        out_fn = out_dir / ("FYKN-" + date_str + ".mp4")

        r = aurorax.data.ucalgary.download("THEMIS_ASI_RAW", start_dt, end_dt, site_uid="fykn")
        data = aurorax.data.ucalgary.read(r.dataset, r.filenames, n_parallel=5)
        images_scaled = at.scale_intensity(data.data, min=1000, max=10000)

        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tqdm.contrib.concurrent import thread_map
        from tqdm import tqdm
        import shutil

        def process_frame(i):
            _, ax = at.display(images_scaled[:, :, i], cmap="gray", returnfig=True)
            ax.text(5, 240, "THEMIS ASI", color="white", size=14)
            ax.text(5, 225, "FYKN", color="white", size=14)
            ax.text(145, 8, data.timestamp[i].strftime("%Y-%m-%d %H:%M:%S UTC"), color="white", size=11)
            filename = "movie_frames/%s_fykn_themis.png" % (data.timestamp[i].strftime("%Y%m%d_%H%M%S"))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=100)
            plt.close("all")
            return filename

        # Use thread_map instead of process_map
        frame_filename_list = thread_map(
            process_frame,
            range(0, images_scaled.shape[-1]),
            max_workers=5,
            desc="Generating frame files: ",
            unit="frames",
        )

        at.movie(frame_filename_list, out_fn, n_parallel=5)

        shutil.rmtree("movie_frames", ignore_errors=True)  # Clean up

    except Exception as e:
        print(f"Error when making fykn video for {date_str}: {e}")


# # Overall

# In[11]:


def get_output_names(date_str, out_dir):
    PKR_video_name = ""
    PKR_keogram_name = ""
    FYKN_video_name = "FYKN-" + date_str + ".mp4"
    FYKN_keogram_name = "FYKN-" + date_str + ".png"

    d = {
        "PKR_video": Path(out_dir / PKR_video_name),
        "PKR_keogram": Path(out_dir / PKR_keogram_name),
        "FYKN_video": Path(out_dir / FYKN_video_name),
        "FYKN_keogram": Path(out_dir / FYKN_keogram_name),
    }

    return d


# In[7]:


def make_output(output_type, date_str, out_dir):
    match output_type:
        case "PKR_video":
            ...
        case "PKR_keogram":
            ...
        case "FYKN_video":
            make_fykn_video(date_str, out_dir)
        case "FYKN_keogram":
            make_fykn_keogram(date_str, out_dir)


# In[19]:

if __name__ == "__main__":
    hst_folder = Path("/Volumes/Elements/organized-outputs")
    out_folder = Path("/Volumes/Elements/THEMIS_outputs")
    out_folder.mkdir(parents=True, exist_ok=True)
    date_folders = [folder for folder in hst_folder.iterdir() if folder.is_dir()]

    for date_folder in date_folders:

        date_str = date_folder.name
        print(f'starting {date_str}')
        year = date_str[:4]

        output_files = get_output_names(date_str, out_folder)

        for output_type, file in output_files.items():
            if file.exists():
                continue
            else:
                print(f'making output for {date_str}')
                make_output(output_type, date_str, out_folder)
