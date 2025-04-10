import pickle
import matplotlib.gridspec as gridspec
import gc
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os
import datetime

# Ghassan's paralell imports
from pathos.multiprocessing import Pool
from functools import partial
import multiprocess.context as ctx

PARALELL_NODES = 16

# # Ghassan Pool example
# def plot_zpara(self, channels, out_dir, nodes):
#     self.calc_ylims()
#     with Pool(nodes) as p:
#         print('Mapping to {} channels'.format(len(channels)))
#         plotter = partial(self.plot_zavg_epoch, out_dir=out_dir)
#         figs = p.map(plotter, channels)

# def transform_hist(linear_interp_by_ch: list,
#                    in_channel_data: list):
    
#     return linear_interp_by_ch[ch_idx](in_channel_data)


def employ_norm(
    files: list,
    file_starts_dt: list,
    num_channels: int,
    file_buffer_sec: float,
    resamp_freq: float,
    save_dir: str,
    scale_type: str,
    scale_factors: list,
    linear_interp_by_ch: list,
    out_dur: float,
    out_stride: float,
    montage: str,
    savename_base: str,
    PROCESS_FILE_DEBUG_LIST: list
    ):
    
    zero_pad_dir = save_dir + '/metadata/zero_padded_epochs'
    if not os.path.exists(zero_pad_dir): os.makedirs(zero_pad_dir)

    total_files = len(files)
    print("\nScaling params aquired: Employing scaling on following files and epoching into desired duration and stride")
    # Re-iterate through all files and normnalize

    # Allow for debugging certain files
    if PROCESS_FILE_DEBUG_LIST:
        process_list = PROCESS_FILE_DEBUG_LIST
    else:
        process_list = range(0,len(files))

    for i in process_list: 
        file = files[i]
        print("\n[" + str(i+1) + '/' + str(total_files) + ']: ' + file)

        # ## Use the previously aquired scale factors to norm the data

        # Load the big pickle
        print("Loading big pickle")
        with open(file, "rb") as f:
            filt_data = pickle.load(f)

        # Ensure that data is in float16 format
        filt_data = np.float16(filt_data)

        scaled_filt_data = np.zeros(filt_data.shape, dtype=np.float16) # initialize for hist scaling below

        # This scale type will be a linear scale with NO zero shift based on percentiles only
        if scale_type=='LinearScale':
            scaled_filt_data = filt_data * scale_factors[:, None]

        elif scale_type == 'CubeRootScale':
            # SCALE BEFORE CubeRoot
            scaled_filt_data = np.cbrt(filt_data * scale_factors[:, None])
        
        elif scale_type == 'HyperTanScaling':
            # SCALE BEFORE TANH
            scaled_filt_data = np.tanh(filt_data * scale_factors[:, None], dtype=np.float16)

        elif scale_type == 'HistEqualScale':
            
            # # Paralellize by mapping channel data to CPU nodes
            # with Pool(PARALELL_NODES) as p:
            #     print('Mapping to {} channels'.format(len(num_channels)))
            #     interpolator = partial(transform_hist, linear_interp_by_ch=linear_interp_by_ch)
            #     scaled_filt_data = p.map(interpolator, channels)
            
            for ch_idx in range(0,num_channels):
                scaled_filt_data[ch_idx,:] = linear_interp_by_ch[ch_idx](filt_data[ch_idx,:])
        
        else:
            raise Exception("scale_type must be 'LinearScale', 'CubeRootScale', 'HyperTanScaling', or HistEqualScale")

        # Clip after scaling
        scaled_filt_data = scaled_filt_data.clip(-1,1)

        # Save output histograms for this file: 
        minISH_file_filtered = np.percentile(filt_data, 0.01, axis=1)
        maxISH_file_filtered = np.percentile(filt_data, 99.99, axis=1)
        
        hist_save_dir = save_dir + '/metadata/normalization_histograms/' + file.split('/')[-1].split('.')[0]
        if not os.path.exists(hist_save_dir): os.makedirs(hist_save_dir)
        # Turn interactive plotting off
        plt.ioff()
        pl.ioff()

        # TAKES A LOT OF MEMORY
        # Save a summary histogram of entire file data points 
        print("Saving histogram for entire file data")
        if np.max(abs(minISH_file_filtered)) > np.max(abs(maxISH_file_filtered)): maxMax = np.max(abs(minISH_file_filtered))
        else: maxMax = np.max(abs(maxISH_file_filtered))
        gs = gridspec.GridSpec(2, 1)
        fig = pl.figure(figsize=(10, 6))
        ax1 = pl.subplot(gs[0, 0])
        ax1.hist(filt_data.flatten(), 200, (-maxMax, maxMax), 'b', alpha = 0.5, label='All Unscaled File Data')
        ax1.title.set_text('Raw Data - All File Data')
        ax1.legend()
        ax2 = pl.subplot(gs[1, 0])
        ax2.hist(filt_data.flatten(), 200, (-1, 1), 'b', alpha = 0.5, label='All Scaled File Data')
        ax2.title.set_text(scale_type + ' - All File Data:')
        ax2.legend()
        if not os.path.exists(f"{hist_save_dir}/JPEGs"): os.makedirs(f"{hist_save_dir}/JPEGs")
        if not os.path.exists(f"{hist_save_dir}/SVGs"): os.makedirs(f"{hist_save_dir}/SVGs")
        savename_jpg = hist_save_dir + '/JPEGs/all_file_data_hist.jpg'
        savename_svg = hist_save_dir + '/SVGs/all_file_data_hist.svg'
        pl.savefig(savename_jpg, dpi=400)
        pl.savefig(savename_svg)
        pl.close(fig)
        del fig

        # Save by channel histograms regardless of channel norm style in order to verify scaling effect
        print("Saving normalization histograms for each channel in file")
        for ch in range(0,num_channels):
            if abs(minISH_file_filtered[ch]) > abs(maxISH_file_filtered[ch]): maxMax_ch  = abs(minISH_file_filtered[ch])
            else: maxMax_ch = abs(maxISH_file_filtered[ch])
            
            gs = gridspec.GridSpec(2, 1)
            fig = pl.figure(figsize=(10, 6))
            ax1 = pl.subplot(gs[0, 0])
            ax1.hist(filt_data[ch,:], 200, (-maxMax_ch, maxMax_ch), 'b', alpha = 0.5, label='All Unscaled File Data for Channel')
            ax1.title.set_text('Raw Data - Channel ID:' + str(ch))
            ax1.legend()
            ax2 = pl.subplot(gs[1, 0])
            ax2.hist(scaled_filt_data[ch,:], 200, (-1, 1), 'b', alpha = 0.5, label='All Scaled File Data for Channel')
            ax2.title.set_text(scale_type + ' - Channel ID:' + str(ch))
            ax2.legend()
            if not os.path.exists(f"{hist_save_dir}/JPEGs"): os.makedirs(f"{hist_save_dir}/JPEGs")
            if not os.path.exists(f"{hist_save_dir}/SVGs"): os.makedirs(f"{hist_save_dir}/SVGs")
            savename_jpg = hist_save_dir + '/JPEGs/' + 'ch' + str(ch) + '.jpg'
            savename_svg = hist_save_dir + '/SVGs/' + 'ch' + str(ch) + '.svg'
            pl.savefig(savename_jpg, dpi=400)
            pl.savefig(savename_svg)
            pl.close(fig)
            del fig

        print("Data scaled, now epoching the file data into Window Duration: " + str(out_dur) + " seconds, Stride: "  + str(out_stride) + " seconds")
        del filt_data
        gc.collect()


        ###### EPOCHING ######
        fs = resamp_freq

        # Now that we have scaled we will epoch into desired duration and stride                      
        end_of_file_datetime = file_starts_dt[i] + datetime.timedelta(seconds=scaled_filt_data.shape[1]/fs)
        stride = datetime.timedelta(seconds=out_stride)
        duration = datetime.timedelta(seconds=out_dur)

        # Skip a buffer time at the beginning and end of the file
        curr_start_datetime = file_starts_dt[i] + datetime.timedelta(seconds=file_buffer_sec) 
        curr_start_sample = int(file_buffer_sec * fs)
        sample_duration = int(out_dur * fs)
        sample_stride = int(out_stride * fs)
        while curr_start_datetime + duration < end_of_file_datetime - datetime.timedelta(seconds=file_buffer_sec):
            # If in loop, then another epoch exists from [start time]  to [start time + window duration]

            # Pull the epoch's data
            epoch_data = np.float16(scaled_filt_data[:,curr_start_sample:(curr_start_sample+sample_duration)])

            if montage == 'BIPOLE': pole_str = '_bipole_'
            if montage == 'MONOPOLE': pole_str = '_monopole_'

            # Pickle the epoch
            # Check for ZERO PADDING and skip epoch if present (probably want to do <small number to account for float16 precision)
            epoch_data_zero_bool = abs(epoch_data) < 1e-7
            if epoch_data_zero_bool.sum() > 1000: # arbitrary check for approximate zeros
                s = "ZERO PADDED FILE FOUND: " + str(out_dur) + " second epoch starting at " + curr_start_datetime.strftime("%m/%d/%Y-%H:%M:%S")
                print(s)
                save_name = zero_pad_dir + "/" + savename_base + "_" + curr_start_datetime.strftime("%m%d%Y_%H%M%S%f")[:-4] + "_to_" + (curr_start_datetime + duration).strftime("%m%d%Y_%H%M%S%f")[:-4] + pole_str + "scaled_filtered_data.pkl"
                
            else: # epoch should be non-zero padded
                save_name = save_dir + "/" + savename_base + "_" + curr_start_datetime.strftime("%m%d%Y_%H%M%S%f")[:-4] + "_to_" + (curr_start_datetime + duration).strftime("%m%d%Y_%H%M%S%f")[:-4] + pole_str + "scaled_filtered_data.pkl"
            
            with open(save_name, "wb") as f:
                pickle.dump(epoch_data, f)

            # Prepare the next round:
            # Advance the window by the stride, but redefine by actual samples to avoid drift
            curr_start_sample = curr_start_sample + sample_stride
            curr_start_datetime = file_starts_dt[i] + datetime.timedelta(seconds=curr_start_sample/fs)

            del epoch_data
            gc.collect()
            
        print("Little pickles saved")

        del scaled_filt_data
        gc.collect()
        print("Garbage collected")