
import os, sys, glob, pickle, numpy as np, random

from utilities import utils_functions, manifold_utilities

if __name__ == "__main__":

    FS = 512 # Currently hardcoded in many places

    # Set to None, otherwise will override all data gathering settings and just run pacmap for the subset of files indicated. 
    subset_override_dir = None # '/media/glommy1/tornados/bse_inference/sheldrake_epoch1138/latent_files/1SecondWindow_1SecondStride'
    subset_override_idxs = [300,301,302,303,304,305]

    # if None, will gather files individually 
    accumulated_data_pickle = None 

    # if 'som_precomputed_path' is None, will train a new SOM, otherwise will run data through pretrained Kohonen weights to make Kohonen map. 
    som_precomputed_path = None # '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/train45/kohonen/64SecondWindow_64SecondStride/all_pats/GPU0_ToroidalSOM_ObjectDict_smoothsec64_Stride64_subsampleFileFactor64_preictalSec14400_gridsize64_lr2with0.9930924954370359decay_sigma32.0with0.9616350847573034decay_numfeatures6645_dims1024_batchsize32_epochs100.pt'  
    
    FS = 512 # Currently hardcoded in many places

    # Master formatted timestamp file - "All Time Data (ATD)"
    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv'
    
    # Which patients to plot
    parent_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/train45'
    # parent_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/bse_inference/val13'
    single_pats = [] # ['Spat18'] # ['Epat27', 'Epat28', 'Epat30', 'Epat31', 'Epat33', 'Epat34', 'Epat35', 'Epat37', 'Epat39', 'Epat41'] # [] # 'Spat18' # 'Spat18' # [] #'Epat35'  # if [] will do all pats 
        
    save_loaded_data = True # Will save one big pickle after all files collected

    # SOURCE DATA
    if (subset_override_dir != None) and (accumulated_data_pickle != None):
        raise Exception('subset_override_dir and accumulated_data_pickle cannot both be not None')
    
    # File subset override
    elif subset_override_dir != None: 
        s = subset_override_dir.split("/")[-1].split("_")
        file_windowseconds = int(s[0].replace("SecondWindow",""))
        file_strideseconds = int(s[1].replace("SecondStride",""))
        
        # Rewindowing data (Must be multiples of original file duration & stride)
        rewin_windowseconds = 1
        rewin_strideseconds = 1

        subsample_file_factor = 1

    # Get metadata from preloaded pickle
    elif accumulated_data_pickle != None: 
        s = accumulated_data_pickle.split("/")[-1].split("_")
        subsample_file_factor = int(s[1].replace("subsampleFileFactor",""))
        rewin_windowseconds = int(s[2].replace("secWindow",""))
        rewin_strideseconds = int(s[3].replace("secStride.pkl",""))

    # Gather raw data
    else: 
        file_windowseconds = 1 # Used for directory construction as a string, so must match directory name exactly
        file_strideseconds = 1
        source_dir = f'{parent_dir}/latent_files/{file_windowseconds}SecondWindow_{file_strideseconds}SecondStride'

        # Rewindowing data (Must be multiples of original file duration & stride)
        rewin_windowseconds = 32
        rewin_strideseconds = 32

        subsample_file_factor = 32 # Not re-windowing, subsampled on a whole file level


    # Plotting Settings
    plot_preictal_color_sec = 60*60*1 #60*60*4
    plot_postictal_color_sec = 0 #60*10 #60*60*4

    # Kohonen Settings [GPU 0]
    som_pca_init = False
    reduction = 'mean' # Keep at mean because currently using reparam in SOM training
    som_device = 0 # GPU
    som_epochs = 100
    som_batch_size = 32
    som_lr = 0.5
    som_lr_min = 0.1 # 0.05
    som_lr_epoch_decay = (som_lr_min / som_lr)**(1 / som_epochs) 
    som_gridsize = 32 
    som_sigma = 0.8 * som_gridsize 
    som_sigma_min = 1.5 # 0.01 * som_gridsize 
    som_sigma_epoch_decay = (som_sigma_min / som_sigma)**(1 / som_epochs) 

    # # Kohonen Settings [GPU 1]
    # som_pca_init = False
    # reduction = 'mean' # Keep at mean because currently using reparam in SOM training
    # som_device = 1 # GPU
    # som_epochs = 100
    # som_batch_size = 32
    # som_lr = 0.5
    # som_lr_min = 0.1
    # som_lr_epoch_decay = (som_lr_min / som_lr)**(1 / som_epochs) 
    # som_gridsize = 64 
    # som_sigma = 0.8 * som_gridsize 
    # som_sigma_min = 1.5 # 0.01 * som_gridsize 
    # som_sigma_epoch_decay = (som_sigma_min / som_sigma)**(1 / som_epochs) 
    
    # Plotting variables
    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code

    # Create paths and directories for saving dim reduction models and outputs
    kohonen_dir = f"{parent_dir}/kohonen/{rewin_windowseconds}SecondWindow_{rewin_strideseconds}SecondStride_Reduction{reduction}"
    if not os.path.exists(kohonen_dir): os.makedirs(kohonen_dir)
    if subset_override_dir != None: kohonen_savedir = f"{kohonen_dir}/file_subset"
    elif single_pats == []: kohonen_savedir = f"{kohonen_dir}/all_pats"
    else: kohonen_savedir = f"{kohonen_dir}/{'_'.join(single_pats)}"
    if not os.path.exists(kohonen_savedir): os.makedirs(kohonen_savedir)


    # ---- END OF PARAMETER SETTING ----


    ### IMPORT DATA ###
    if (subset_override_dir != None) or (accumulated_data_pickle == None): # If override files or gathering raw files
        
        if subset_override_dir != None: # File subset override
            print("FILE SUBSET OVERRIDE")
            all_dir_files = glob.glob(f"{subset_override_dir}/*.pkl")
            if len(all_dir_files) == 0:
                raise Exception("ERROR: no files found")
            if len(subset_override_idxs) == 1:
                data_filepaths = [all_dir_files[subset_override_idxs[0]]]
            else:
                data_filepaths = [all_dir_files[i] for i in subset_override_idxs]

        else: # Gather data from files
            print("No prelaoded data pickle supplied, gathering files seperately")
            print(f"Source directory: {source_dir}")
            data_filepaths = [] 
            if single_pats == []: data_filepaths = data_filepaths + glob.glob(source_dir + f'/*.pkl')
            else: # Get all of the patients listing
                num_single_pats = len(single_pats)
                for j in range(num_single_pats):
                    pat_curr = single_pats[j]
                    data_filepaths = data_filepaths + glob.glob(source_dir + f'/{single_pats[j]}*.pkl')

            # Subsample files (to fit in RAM)
            if subsample_file_factor > 1:
                print(f"Subsampling files by {subsample_file_factor}")
                random.shuffle(data_filepaths)
                data_filepaths = data_filepaths[::subsample_file_factor]
            else:
                print("No file subsampling")

            assert (data_filepaths[0].split("/")[-1].split("_")[-1] == f"{file_strideseconds}secStride.pkl") & (data_filepaths[0].split("/")[-1].split("_")[-2] == f"{file_windowseconds}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
        
        # Gather metadata for files
        build_start_datetimes, build_stop_datetimes = manifold_utilities.filename_to_datetimes([s.split("/")[-1] for s in data_filepaths]) # Get start/stop datetimes
        build_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in data_filepaths] # Get the build pat_ids
        
        # Load a sentinal file to get data params and intitialzie variables
        with open(data_filepaths[0], "rb") as f: latent_data_fromfile = pickle.load(f)
        ww_means_sentinel = latent_data_fromfile['windowed_weighted_means']
        ww_logvars_sentinel = latent_data_fromfile['windowed_weighted_logvars']
        w_mogpreds_sentinel = latent_data_fromfile['windowed_mogpreds']
        print(f"Original shape of ww_means: {ww_means_sentinel.shape}")
        print(f"Original shape of ww_logvars: {ww_logvars_sentinel.shape}")
        print(f"Original shape of w_mogpreds: {w_mogpreds_sentinel.shape}")
        if (file_windowseconds != rewin_windowseconds) or (file_strideseconds != rewin_strideseconds):
            if (file_windowseconds > rewin_windowseconds) or (file_strideseconds > rewin_strideseconds):
                raise Exception("ERROR: desired window duration or stride is smaller than raw file duration or stride")
            else:
                rewin_means_sentinel, rewin_logvars_sentinel, rewin_mogpreds_sentinel = utils_functions.rewindow_data(
                    ww_means_sentinel, ww_logvars_sentinel, w_mogpreds_sentinel, 
                    file_windowseconds, file_strideseconds, rewin_windowseconds, rewin_strideseconds,
                    reduction=reduction)
                print(f"Rewindowed shape of rewin_means: {rewin_means_sentinel.shape}")
                print(f"Rewindowed shape of rewin_logvars: {rewin_logvars_sentinel.shape}")
                print(f"Rewindowed shape of rewin_mogpreds: {rewin_mogpreds_sentinel.shape}")
                print(f"REDUCTION: {reduction}")
        else:
            rewin_means_sentinel = ww_means_sentinel
            rewin_logvars_sentinel = ww_logvars_sentinel
            rewin_mogpreds_sentinel = w_mogpreds_sentinel
            print("Data NOT re-windowed")

        # Inialize all_file variables based on sentinel variables
        ww_means_allfiles = np.zeros([len(data_filepaths), rewin_means_sentinel.shape[0], rewin_means_sentinel.shape[1]], dtype=np.float32)
        ww_logvars_allfiles = np.zeros([len(data_filepaths), rewin_logvars_sentinel.shape[0], rewin_logvars_sentinel.shape[1]], dtype=np.float32)
        w_mogpreds_allfiles = np.zeros([len(data_filepaths), rewin_mogpreds_sentinel.shape[0], rewin_mogpreds_sentinel.shape[1]], dtype=np.float32)
        print("Loading all BUILD latent data from files")
        for i in range(len(data_filepaths)):
            sys.stdout.write(f"\rLoading Pickles: {i}/{len(data_filepaths)}        ") 
            sys.stdout.flush() 
            try:
                with open(data_filepaths[i], "rb") as f: latent_data_fromfile = pickle.load(f)
                ww_means = latent_data_fromfile['windowed_weighted_means']
                ww_logvars = latent_data_fromfile['windowed_weighted_logvars']
                w_mogpreds = latent_data_fromfile['windowed_mogpreds']
                if (file_windowseconds != rewin_windowseconds) or (file_strideseconds != rewin_strideseconds):
                    ww_means_allfiles[i, :, :], ww_logvars_allfiles[i, :, :], w_mogpreds_allfiles[i, :, :] = utils_functions.rewindow_data(
                        ww_means, ww_logvars, w_mogpreds, file_windowseconds, file_strideseconds, rewin_windowseconds, rewin_strideseconds, reduction=reduction)
                else: # No re-windowing needed
                    ww_means_allfiles[i, :, :] = ww_means
                    ww_logvars_allfiles[i, :, :] = ww_logvars
                    w_mogpreds_allfiles[i, :, :] = w_mogpreds                
            except: print(f"Error loading {data_filepaths[i]}")

        # Save the gathered data to save time for re-plotting
        if save_loaded_data and ((subset_override_dir == None) and (accumulated_data_pickle == None)):
            savepath = f"{kohonen_savedir}/allDataGathered_subsampleFileFactor{subsample_file_factor}_{rewin_windowseconds}secWindow_{rewin_strideseconds}secStride.pkl"
            output_obj = open(savepath, 'wb')
            save_dict = {
                'ww_means_allfiles': ww_means_allfiles,
                'ww_logvars_allfiles': ww_logvars_allfiles,
                'w_mogpreds_allfiles': w_mogpreds_allfiles,
                'build_start_datetimes': build_start_datetimes,
                'build_stop_datetimes': build_stop_datetimes,
                'build_pat_ids_list': build_pat_ids_list}
            pickle.dump(save_dict, output_obj)
            output_obj.close()
            print(f"Gathered data saved to one big pickle: {savepath}")

    else: # Get metadata from preloaded pickle
        print("Preloaded single data pickle supplied for all data")
        with open(accumulated_data_pickle, "rb") as f: data_loaded = pickle.load(f)
        ww_means_allfiles = data_loaded['ww_means_allfiles']
        ww_logvars_allfiles = data_loaded['ww_logvars_allfiles']
        w_mogpreds_allfiles = data_loaded['w_mogpreds_allfiles']
        build_start_datetimes = data_loaded['build_start_datetimes']
        build_stop_datetimes = data_loaded['build_stop_datetimes']
        build_pat_ids_list = data_loaded['build_pat_ids_list']

    # Print broad statistics about dataset
    print(f"Dataset general statistics:\n"
          f"Means: Mean {np.mean(ww_means_allfiles):.2f}\n"
          f"Means: Std {np.std(ww_means_allfiles):.2f}\n"
          f"Means: Max {np.max(ww_means_allfiles):.2f}\n"
          f"Means: Min {np.min(ww_means_allfiles):.2f}\n"
          f"Logvars: Mean {np.mean(ww_logvars_allfiles):.2f}\n"
          f"Logvars: Std {np.std(ww_logvars_allfiles):.2f}\n"
          f"Logvars: Max {np.max(ww_logvars_allfiles):.2f}\n"
          f"Logvars: Min {np.min(ww_logvars_allfiles):.2f}"
          )

    manifold_utilities.toroidal_kohonen_subfunction_pytorch(
        atd_file = atd_file,
        pat_ids_list=build_pat_ids_list,
        latent_means_windowed=ww_means_allfiles,
        latent_logvars_windowed=ww_logvars_allfiles,
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        win_sec=rewin_windowseconds, 
        stride_sec=rewin_strideseconds, 
        savedir=kohonen_savedir,
        subsample_file_factor=subsample_file_factor,
        som_pca_init=som_pca_init,
        som_precomputed_path=som_precomputed_path,
        som_device=som_device,
        som_batch_size=som_batch_size,
        som_lr=som_lr,
        som_epochs=som_epochs,
        som_gridsize=som_gridsize,
        som_lr_epoch_decay=som_lr_epoch_decay,
        som_sigma=som_sigma,
        som_sigma_epoch_decay=som_sigma_epoch_decay,
        som_sigma_min=som_sigma_min,
        FS = FS,
        plot_preictal_color_sec = plot_preictal_color_sec,
        plot_postictal_color_sec = plot_postictal_color_sec,
        **kwargs)
