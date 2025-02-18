import os
import shutil
import glob
import datetime
from utilities import utils_functions
import pandas as pd
import numpy as np
import pickle
import gc
from scipy import interpolate

def find_contiguous_true_indexes(array):
  """Finds the indices of all contiguous regions of True in an array.

  Args:
    array: A NumPy boolean array.

  Returns:
    A list of tuples, where each tuple contains the start and end indices of a
    contiguous region of True values.
  """

  # Find the indices of all True values in the array.
  true_indexes = np.nonzero(array)[0]

  # Find the start and end indices of each contiguous region of True values.
  start_indexes = []
  end_indexes = []
  for i in range(len(true_indexes)):
    if i == 0 or true_indexes[i] - true_indexes[i - 1] > 1:
      start_indexes.append(true_indexes[i])
    if i == len(true_indexes) - 1 or true_indexes[i + 1] - true_indexes[i] > 1:
      end_indexes.append(true_indexes[i])

  # Return a list of tuples containing the start and end indices of each
  # contiguous region of True values.
  return list(zip(start_indexes, end_indexes))


def aquire_scale_params(pat_id: str,
                        num_channels: int,
                        big_pickle_dir: str,
                        files: list,
                        file_starts_dt: list,
                        atd_file: str,
                        save_dir: str,
                        scale_epoch_type: str,
                        scale_epoch_hours: float,
                        buffer_start_hours: float,
                        channel_scale_style: str,
                        scale_type: str,
                        resamp_freq: float,
                        histo_min: float,
                        histo_max: float,
                        num_bins: int,
                        ):    

    scale_factors = [] # WIll output for all non-hist strategies
    linear_interp_by_ch = [] # Will output for hist normalization

    # Get the first tfile datetime, but SORT first
    all_start_seconds = [int((x - x.min).total_seconds()) for x in file_starts_dt]
    sort_idxs = np.argsort(all_start_seconds)
    sorted_dts = [file_starts_dt[i] for i in sort_idxs]
    first_file_start_datetime = sorted_dts[0]

    print("\nFiles will be processed with following selections:")

    if scale_epoch_type == 'data_normalized_to_first':
        print("data_normalized_to_first X hours")
        normalization_epoch_sec = [buffer_start_hours*3600, (buffer_start_hours + scale_epoch_hours)*3600] # seconds, range in which to take the normalization epoch from first EDF file


    elif scale_epoch_type == 'data_normalized_to_first_seizure_centered':
        print("data_normalized_to_first_seizure_centered")
        # Get the seizure datetimes for this patient
        seiz_start_datetimes, seiz_stop_datetimes, seiz_types = utils_functions.get_pat_seiz_datetimes(
            pat_id=pat_id, 
            atd_file=atd_file,                           
            FBTC_bool=True, 
            FIAS_bool=True, 
            FAS_to_FIAS_bool=True,
            FAS_bool=True, 
            subclinical_bool=False, 
            unknown_bool=False,  ################################
            non_electro_bool=False)

        # If seizure is too close to beginning, then start normalizatiomn at beginning of EMU stay
        seiz_1_start = seiz_start_datetimes[0]
        if seiz_1_start - first_file_start_datetime < datetime.timedelta(hours=scale_epoch_hours/2):
            normalization_epoch_sec = [buffer_start_hours * 3600, (buffer_start_hours + scale_epoch_hours)*3600]

        # There is enough data to before first seizure to center the normalization epochs around the first seizure
        else:
            normalization_epoch_sec = [((seiz_1_start - datetime.timedelta(hours=scale_epoch_hours/2)) - first_file_start_datetime).total_seconds(),
                                    ((seiz_1_start + datetime.timedelta(hours=scale_epoch_hours/2)) - first_file_start_datetime).total_seconds()]


    else:
        raise Exception("'scale_epoch_type' must equal 'data_normalized_to_first_seizure_centered' or 'data_normalized_to_first'")
        
        
    # Save the normalization epoch seconds
    df_norm_epoch = pd.DataFrame({'first_file_start_datetime': first_file_start_datetime,
                                'normalization_epoch_start_sec': normalization_epoch_sec[0],
                                'normalization_epoch_end_sec': normalization_epoch_sec[1]}, index=[0])
    if not os.path.exists(save_dir + '/metadata/scaling_metadata/'): os.mkdir(save_dir + '/metadata/scaling_metadata/')
    df_norm_epoch.to_csv(save_dir + '/metadata/scaling_metadata/' + 'normalization_epoch_seconds.csv')


    # ### AQUIRE NORMALIZE VALUES

    # First, aquire the normalization parameters. 
    # Do this is a seperate loop through the files because 
    # it may take more than one file to get enough data for desired normalization range
    scale_factors =  np.ones(num_channels, dtype=np.float16)
    histo_bin_edges = np.linspace(histo_min,histo_max,num_bins+1)
    histo_bin_counts = np.zeros([num_channels, num_bins])

    print(channel_scale_style)
    print(scale_type)

    file_idx = -1
    total_files = len(files)
    print("\nFollowing files COULD contain normalization range of " + str(normalization_epoch_sec[0]/3600) + "-" + str(normalization_epoch_sec[1]/3600) + " hours from start of EMU stay:")
    for i in range(0,len(files)):
        gc.collect()
        file_idx += 1
        file = files[i]

        # Check if this file COULD contain normalization data in range of interest
        # Problem is we do NOT have end timestamps for EMU files, so hard to know if a particular file has data in desired range
        if (file_starts_dt[i] < first_file_start_datetime + datetime.timedelta(seconds=normalization_epoch_sec[1])):
            
            print("[" + str(file_idx+1) + '/' + str(total_files) + ']: ' + file)
            # Load the big pickle
            with open(file, "rb") as f:
                filt_data_norm = pickle.load(f)

            # Ensure data is float 16
            filt_data_norm = np.float16(filt_data_norm)
        
            seconds_in_file = filt_data_norm.shape[1]/resamp_freq 

            # Determine if only partial data from this file is needed (i.e. within the normalization epoch range)
            file_end_dt = file_starts_dt[i] + datetime.timedelta(seconds=seconds_in_file)
            norm_end_dt = first_file_start_datetime + datetime.timedelta(seconds=normalization_epoch_sec[1])
            norm_start_dt = first_file_start_datetime + + datetime.timedelta(seconds=normalization_epoch_sec[0])
            
            # Skip this file if it does not contain any data in normalization range
            if file_end_dt < norm_start_dt:
                print("No data in normalization range, skipping file")
                continue

            # Only need later portion of file
            elif  (file_starts_dt[i] <= norm_start_dt) & (file_end_dt <= norm_end_dt):
                start_samp_needed_idx = (file_end_dt - norm_start_dt).seconds * resamp_freq
                filt_data_norm = filt_data_norm[:, start_samp_needed_idx:]

            # Only need middle portion of file
            elif  (file_starts_dt[i] <= norm_start_dt) & (file_end_dt > norm_end_dt):
                start_samp_needed_idx = (file_end_dt - norm_start_dt).seconds * resamp_freq
                end_samp_needed_idx = (norm_end_dt - file_starts_dt[i]).seconds * resamp_freq
                filt_data_norm = filt_data_norm[:, start_samp_needed_idx:end_samp_needed_idx]
            
            # Only need first portion of file
            elif  (file_starts_dt[i] > norm_end_dt) & (file_end_dt > norm_end_dt):
                end_samp_needed_idx = (norm_end_dt - file_starts_dt[i]).seconds * resamp_freq
                filt_data_norm = filt_data_norm[:, 0:end_samp_needed_idx]


            # Check for ZERO PADDING and skip epoch if present (probably want to do <small number to account for float16 precision)
            epoch_data_zero_bool = abs(filt_data_norm) < 1e-7
            # Iterate to find islands of zeros, then delete        
            true_islands = find_contiguous_true_indexes(epoch_data_zero_bool[0, :])
            zero_island_delete_idxs = []
            for ir in reversed(range(len(true_islands))):
                island = true_islands[ir]
                if ((island[1] - island[0]) > resamp_freq):
                    # raise Exception("NEED TO CODE UP ZERO ISLAND DELETION")
                    zero_island_delete_idxs = zero_island_delete_idxs + [island]
                
            # Modify based on channel norming style
            if channel_scale_style =='Same_Scale_For_All_Channels':
                
                if scale_type == 'HistEqualScale':
                    new_histo_bin_counts = utils_functions.fill_hist_by_channel(data_in=filt_data_norm, histo_bin_edges=histo_bin_edges, zero_island_delete_idxs=zero_island_delete_idxs)
                    # now sum along channel because using the same scale for all channels
                    new_histo_bin_counts[:,:] = np.sum(new_histo_bin_counts, axis=0)
                    histo_bin_counts = histo_bin_counts + new_histo_bin_counts

                else:
                    # Scale data to be used by norm paradigms
                    val99 = np.percentile(filt_data_norm, 99)
                    val1 = np.percentile(filt_data_norm, 1)
                    scale_factors_curr = np.zeros(filt_data_norm.shape[0], dtype=np.float16)
                    scale_factors_curr[:] = (1/(val99-val1)).astype(np.float16)
                    for i in range(len(scale_factors_curr)):
                        # Check if scale needs to be shrunk according to a new larger range found
                        if scale_factors_curr[i] < scale_factors[i]:
                            scale_factors[i] = scale_factors_curr[i]

            elif channel_scale_style == 'By_Channel_Scale':

                if scale_type == 'HistEqualScale':
                    histo_bin_counts = histo_bin_counts + utils_functions.fill_hist_by_channel(data_in=filt_data_norm, histo_bin_edges=histo_bin_edges, zero_island_delete_idxs=zero_island_delete_idxs)

                else:
                    # Scale data to be used by norm paradigms
                    val99 = np.percentile(filt_data_norm, 99, axis=1)
                    val1 = np.percentile(filt_data_norm, 1, axis=1)
                    scale_factors_curr = (1/(val99-val1)).astype(np.float16)
                    for i in range(len(scale_factors_curr)):
                        # Check if scale needs to be shrunk according to a new larger range found
                        if scale_factors_curr[i] < scale_factors[i]:
                            scale_factors[i] = scale_factors_curr[i]

            else:
                raise Exception("'channel_scale_style' must be 'Same_Scale_For_All_Channels', 'By_Channel_Scale' of 'HistEqualScale")


    # Save the scaling data
    if not os.path.exists(save_dir + '/metadata/scaling_metadata/'): os.mkdir(save_dir + '/metadata/scaling_metadata/')

    # Save the histogram values 
    if scale_type == 'HistEqualScale':
        # Save a CSV and Pickle to output directory  
        df_hist_counts = pd.DataFrame(histo_bin_counts, columns=histo_bin_edges[0:-1])
        df_hist_counts.to_csv(save_dir + '/metadata/scaling_metadata/' + 'histo_bin_counts.csv')
        save_name = save_dir + '/metadata/scaling_metadata/' + 'histo_bin_counts.pkl'
        with open(save_name, "wb") as f:
            pickle.dump(df_hist_counts, f)

        # Get the CDF for the channel histograms
        cdf_x_avg_vals = (histo_bin_edges[0:-1] + histo_bin_edges[1:])/2
        cdf_by_channel = np.zeros([num_channels, num_bins])
        cdf_by_channel_scaled = np.zeros([num_channels, num_bins])
        
        # Iterate through channels
        for ch_idx in range(0,num_channels):
            cdf_by_channel[ch_idx,0] = histo_bin_counts[ch_idx,0] # initialize the first bin
            print(f"Channel ID: {str(ch_idx)} --> TODO list commprehension with subprocess")

            # TODO list comprehension
            # cdf_by_channel[ch_idx, :] = [scale_subprocess(bin_idx, ch_idx) for bin_idx in range(1,num_bins)]

            for bin_idx in range(1,num_bins):
                cdf_by_channel[ch_idx, bin_idx] = cdf_by_channel[ch_idx, bin_idx -1] + histo_bin_counts[ch_idx,bin_idx]
            
                # Rescale the CDF [-1, 1]
                # Ensure that the x-axis bin that contains ZERO lines up with the Y-axis ZERO
                # Do this by scaling the top half and bottom half of CDF individually
                # TODO? do not want a slope discontinuity at zero, thus smooth the CDF scaling values toward zero
                max_val = cdf_by_channel[ch_idx, num_bins-1]
                zero_bin_val = cdf_by_channel[ch_idx, int((num_bins-1)/2)]
                min_val = cdf_by_channel[ch_idx, 0]

                # Shift the whole curve down to be centered at zero
                cdf_by_channel_scaled[ch_idx, :] = cdf_by_channel[ch_idx, :] - zero_bin_val

                # Scale each half of the curve
                cdf_by_channel_scaled[ch_idx, 0 : int((num_bins-1)/2)] = cdf_by_channel_scaled[ch_idx, 0 : int((num_bins-1)/2)] / (zero_bin_val - min_val)
                cdf_by_channel_scaled[ch_idx, int((num_bins-1)/2) + 1 : ] = cdf_by_channel_scaled[ch_idx, int((num_bins-1)/2) + 1 : ] / (max_val - zero_bin_val)

        # Fit a linear interpolation function to all scaled channel CDFs individually (will just be the same if scaling the same across channels)
        linear_interp_by_ch = [interpolate.interp1d(cdf_x_avg_vals, cdf_by_channel_scaled[ch_idx, :], kind='linear', bounds_error=False, fill_value=(-1, 1)) for ch_idx in range(num_channels)]

        # Pickle the interpolations
        save_name = save_dir + '/metadata/scaling_metadata/' + 'linear_interpolations_by_channel.pkl'
        with open(save_name, "wb") as f:
            pickle.dump(linear_interp_by_ch, f)


    else: # For all other scale types, save the scale values
        # Save a CSV and Pickle to output directory  
        df_scale_factors = pd.DataFrame({'chan_scale_vals': scale_factors})
        df_scale_factors.to_csv(save_dir + '/metadata/scaling_metadata/' + 'channel_scale_factors.csv')
        save_name = save_dir + '/metadata/scaling_metadata/' + 'channel_scale_factors.pkl'
        with open(save_name, "wb") as f:
            pickle.dump(df_scale_factors, f)

    # Clear up space for next file loop
    del filt_data_norm
    gc.collect()

    return scale_factors, linear_interp_by_ch