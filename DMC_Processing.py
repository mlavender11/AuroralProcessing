#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import importlib
import hdf_utils
from tqdm.auto import tqdm
# sys.path.append('/Users/michaellavender/Documents/BUSPC/batch_processing')
from binary_to_hdf import whole_binary_to_hdf


# In[ ]:


drive_path = Path('/Volumes/I')


# In[ ]:


DMC_folders = []
for item in drive_path.iterdir():
    if item.is_dir():
        if any(item.glob('*.DMCdata')):
            DMC_folders.append(item)


# In[ ]:


DMC_files = []
for folder in DMC_folders:
    for DMC_file in folder.glob('*.DMCdata'):
        DMC_files.append(DMC_file)


# In[ ]:


# DMC_folders


# In[ ]:


# DMC_files


# In[ ]:


DMC_files_cleaned = [file for file in DMC_files if not ('frames' in file.name or 'test' in file.name or '2012' in file.name)]


# In[ ]:


# DMC_files_cleaned


# In[ ]:


from histutils.convert.__main__ import convert_DMC_to_hdf5
# importlib.reload(histutils.convert.__main__)


# In[16]:


# Looping
out_dir = Path('/Volumes/Elements/DMC_output')
out_dir.mkdir(parents=True, exist_ok=True)

for DMC_fn in DMC_files_cleaned:
    print(DMC_fn)
    try:
        out_fn = out_dir / DMC_fn.with_suffix('.h5').name
        log_fn = DMC_fn.with_suffix('.log')
        if not log_fn.exists():
            print(f"Skipping: Log file not found for {DMC_fn} (expected {log_fn})")
            print('.' * 40 + '\n')
            continue

        start_time = hdf_utils.parse_log_for_start_time(log_fn)
        print(f'start time: {start_time}')

        convert_DMC_to_hdf5(DMC_fn, out_fn, {"header_bytes": 4, "startUTC": start_time})
        print('.' * 40 + '\n')

    except Exception as e:
        print(f'Error in file {DMC_fn}: {e}')


# In[ ]:




