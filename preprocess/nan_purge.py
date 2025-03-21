"""
This script scans directories containing `.pkl` files within a given parent directory,
searches for files containing NaN (Not a Number) values, and provides the option to delete 
those files by moving them to a designated delete directory.

The script operates on a directory structure where each subdirectory contains `.pkl` files. 
It loads each file, checks for the presence of NaN values in the contents, and if any NaNs 
are found, the user is prompted for confirmation to delete those files. The files marked for 
deletion are moved to a separate directory (`delete_directory`), and a log of deleted files 
is saved in the original directory.

### Attributes:
-------------
- `parent_dir` : str
    The path to the parent directory where patient data directories are located. 
    Each subdirectory within this parent directory is scanned for `.pkl` files.
  
- `dir_keyphrase` : str
    A keyphrase used to identify subdirectories related to patient data (typically "pat" in the example).

- `delete_directory` : str
    The path where `.pkl` files containing NaN values will be moved after deletion confirmation.
    A new directory is created if it does not exist.

### Process:
---------
1. The script identifies directories containing patient data files by searching for 
   directories that match the `dir_keyphrase` within the `parent_dir`.
   
2. For each identified directory, the script loads every `.pkl` file and checks for NaN values 
   using `np.isnan()`. If a NaN is found, the file path is added to the deletion list.

3. The user is prompted to confirm whether they would like to delete the files containing NaNs. 
   If confirmed ('y'), the files are moved to the `delete_directory`.

4. A log of deleted files is saved in the respective directory in a text file called 
   `files_deleted_with_nans.txt`.

"""

import glob
import pickle
import numpy as np
import shutil
import os

parent_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats'
dir_keyphrase = 'pat'

delete_directory = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats/deleted_nan_files'
if not os.path.exists(delete_directory): os.makedirs(delete_directory)

pat_dirs = glob.glob(f"{parent_dir}/*{dir_keyphrase}*")

for i in range(len(pat_dirs)):
    dir_curr = pat_dirs[i]
    files_curr = glob.glob(f"{dir_curr}/*/*/*.pkl")
    print(f"Reading: {dir_curr}\n{len(files_curr)} files")

    # TODO parllelize

    del_list = []
    for j in range(len(files_curr)):
        with open(files_curr[j], "rb") as f: latent_data_windowed = pickle.load(f)
        if np.isnan(latent_data_windowed).any():
            print(f"Found NaNs: {files_curr[j]}")
            del_list.append(files_curr[j])
    
    if del_list != []:
        usr_agree = input(f"Delete list:\n {del_list}\n\n Continue (y/n): ")
        while (usr_agree != 'y') and (usr_agree != 'n'):
            usr_agree = input(f"Delete list:\n {del_list}\n\n Continue (y/n) - Must input 'y' or 'n: ")

        # Move files to delete directory and leave log of deleted files
        if usr_agree == 'y':
            for j in range(len(del_list)):
                filename = del_list[j].split("/")[-1]
                shutil.move(del_list[j], f"{delete_directory}/{filename}")

            # Save a log of what was moved
            with open(f"{dir_curr}/files_deleted_with_nans.txt", 'w') as file:
                file.writelines(line.split("/")[-1] + '\n' for line in del_list)

    else:
        print("No NaNs found")

    
