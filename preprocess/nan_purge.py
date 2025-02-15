'''
Delete any file with NaNs
Leave a log of deleted files
'''

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

    
