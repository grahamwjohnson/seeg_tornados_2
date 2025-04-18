
import os, sys, glob, pickle, numpy as np

from utilities import utils_functions, manifold_utilities

if __name__ == "__main__":

    FS = 512 # Currently hardcoded in many places

    # Master formatted timestamp file - "All Time Data (ATD)"
    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/pangolin_ripple/trained_models/all_time_data_01092023_112957.csv'

    # Source data selection
    file_windowsecs = 64 # Used for directory construction, so must match directory name exactly
    file_stridesecs = 64
    parent_dir = f'/media/glommy1/tornados/bse_inference/sheldrake_epoch1138'
    source_dir = f'{parent_dir}/latent_files/{file_windowsecs}SecondWindow_{file_stridesecs}SecondStride'
    single_pats = ['Spat18'] # ['Epat27', 'Epat28', 'Epat30', 'Epat31', 'Epat33', 'Epat34', 'Epat35', 'Epat37', 'Epat39', 'Epat41'] # [] # 'Spat18' # 'Spat18' # [] #'Epat35'  # if [] will do all pats 
    
    # Rewindowing data
    rewin_windowsecs = 64
    rewin_strideseconds = 64

    # HDBSCAN Settings
    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100

    # Plotting Settings
    plot_preictal_color_sec = 60*60*4 #60*60*4
    plot_postictal_color_sec = 0 #60*10 #60*60*4

    # Kohonen Settings
    # if 'som_precomputed_path' is None, will train a new SOM, otherwise will run data through pretrained Kohonen weights to make Kohonen map. 
    som_precomputed_path = None # '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats/trained_models/dataset_train90.0_val10.0/tmp_incatern/kohonen/Epoch33/64SecondWindow_64SecondStride/all_pats/generation/som_state_dict.pt'
    som_device = 0 # GPU
    som_batch_size = 256
    som_lr = 0.5
    som_lr_epoch_decay = 0.995
    som_epochs = 1000
    som_gridsize = 25
    som_sigma = 15 
    som_sigma_epoch_decay = 0.995
    som_sigma_min = 0.1

    # Plotting variables
    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code

    # Create paths and directories for saving dim reduction models and outputs
    kohonen_dir = f"{parent_dir}/kohonen/{rewin_windowsecs}SecondWindow_{rewin_strideseconds}SecondStride"
    if not os.path.exists(kohonen_dir): os.makedirs(kohonen_dir)


     ### IMPORT DATA ###
    data_filepaths = [] 
    if single_pats == []: data_filepaths = data_filepaths + glob.glob(source_dir + f'/*.pkl')
    else: # Get all of the patients listing
        num_single_pats = len(single_pats)
        for j in range(num_single_pats):
            pat_curr = single_pats[j]
            data_filepaths = data_filepaths + glob.glob(source_dir + f'/{single_pats[j]}*.pkl')
    
    assert (data_filepaths[0].split("/")[-1].split("_")[-1] == f"{file_stridesecs}secStride.pkl") & (data_filepaths[0].split("/")[-1].split("_")[-2] == f"{file_windowsecs}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
    build_start_datetimes, build_stop_datetimes = manifold_utilities.filename_to_datetimes([s.split("/")[-1] for s in data_filepaths]) # Get start/stop datetimes
    build_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in data_filepaths] # Get the build pat_ids
    
    # Load a sentinal file to get data params and intitialzie variables
    with open(data_filepaths[0], "rb") as f: latent_data_fromfile = pickle.load(f)
    ww_means_sentinel = latent_data_fromfile['windowed_weighted_means']
    ww_logvars_sentinel = latent_data_fromfile['windowed_weighted_logvars']
    w_mogpreds_sentinel = latent_data_fromfile['windowed_mogpreds']
    ww_means_allfiles = np.zeros([len(data_filepaths), ww_means_sentinel.shape[0], ww_means_sentinel.shape[1]], dtype=np.float32)
    ww_logvars_allfiles = np.zeros([len(data_filepaths), ww_logvars_sentinel.shape[0], ww_logvars_sentinel.shape[1]], dtype=np.float32)
    w_mogpreds_allfiles = np.zeros([len(data_filepaths), w_mogpreds_sentinel.shape[0], w_mogpreds_sentinel.shape[1]], dtype=np.float32)
    print("Loading all BUILD latent data from files")
    latent_data_fromfile = [''] * len(data_filepaths) # Load all of the data into system RAM - list of [window, latent_dim]
    for i in range(len(data_filepaths)):
        with open(data_filepaths[i], "rb") as f: latent_data_fromfile = pickle.load(f)
        ww_means_allfiles[i, :, :] = latent_data_fromfile['windowed_weighted_means']
        ww_logvars_allfiles[i, :, :] = latent_data_fromfile['windowed_weighted_logvars']
        w_mogpreds_allfiles[i, :, :] = latent_data_fromfile['windowed_mogpreds']

    # Rewindow the data if needed
    if (file_windowsecs != rewin_windowsecs) or (file_stridesecs != rewin_strideseconds):
        if (file_windowsecs > rewin_windowsecs) or (file_stridesecs > rewin_strideseconds):
            raise Exception("ERROR: desired window duration or stride is smaller than raw file duration or stride")
        else:
            print(f"Original shape of ww_means: {ww_means_allfiles.shape}")
            print(f"Original shape of ww_logvars: {ww_logvars_allfiles.shape}")
            print(f"Original shape of w_mogpreds: {w_mogpreds_allfiles.shape}")
            rewin_means_allfiles, rewin_logvars_allfiles, rewin_mogpreds_allfiles = utils_functions.rewindow_data_filewise(
                ww_means_allfiles, ww_logvars_allfiles, w_mogpreds_allfiles, file_windowsecs, file_stridesecs, rewin_windowsecs, rewin_strideseconds)
            print(f"Rewindowed shape of rewin_means: {rewin_means_allfiles.shape}")
            print(f"Rewindowed shape of rewin_logvars: {rewin_logvars_allfiles.shape}")
            print(f"Rewindowed shape of rewin_mogpreds: {rewin_mogpreds_allfiles.shape}")
    else:
        rewin_means_allfiles = ww_means_allfiles
        rewin_logvars_allfiles = ww_logvars_allfiles
        rewin_mogpreds_allfiles = w_mogpreds_allfiles
        print("Data NOT re-windowed")

    ### Run the Kohonen / Self Organizing Maps (SOM) Algorithm
    if single_pats == []: kohonen_savedir = f"{kohonen_dir}/all_pats/generation"
    else: kohonen_savedir = f"{kohonen_dir}/{'_'.join(single_pats)}/generation"
    axes, som = manifold_utilities.kohonen_subfunction_pytorch(
        atd_file = atd_file,
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=rewin_means_allfiles, # Just the means for kohonen
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        win_sec=rewin_windowsecs, 
        stride_sec=rewin_strideseconds, 
        savedir=kohonen_savedir,
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
        HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
        HDBSCAN_min_samples = HDBSCAN_min_samples,
        plot_preictal_color_sec = plot_preictal_color_sec,
        plot_postictal_color_sec = plot_postictal_color_sec,
        **kwargs)
