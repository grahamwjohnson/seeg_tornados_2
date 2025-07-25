import os, glob
import pickle
from  utilities import manifold_utilities
import numpy as np
import random

'''
@author: grahamwjohnson
June 2025

Ad-hoc script to run PaCMAP/Histograms on BSV latent files
'''

import os, sys, glob, pickle, numpy as np

from utilities import utils_functions, manifold_utilities

if __name__ == "__main__":

    run_histos = False
    run_pacmap = True
    run_phate = True

    # Set to None, otherwise will override all data gathering settings and just run pacmap for the subset of files indicated. 
    subset_override_dir = None # '/media/glommy1/tornados/bse_inference/sheldrake_epoch1138/latent_files/1SecondWindow_1SecondStride'
    subset_override_idxs = [300,301,302,303,304,305]

    # if None, will gather files individually 
    accumulated_data_pickle = None # '/media/glommy1/tornados/bse_inference/train45/pacmap/256SecondWindow_256SecondStride/all_pats/allDataGathered_256secWindow_256secStride.pkl'  
    
    # Which patients to plot
    # parent_dir = f'/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_train45'
    # parent_dir = f'/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_val13'
    parent_dir = f'/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_all58'
    single_pats = [] # ['Spat18'] # ['Epat27', 'Epat28', 'Epat30', 'Epat31', 'Epat33', 'Epat34', 'Epat35', 'Epat37', 'Epat39', 'Epat41'] # [] # 'Spat18' # 'Spat18' # [] #'Epat35'  # if [] will do all pats 

    # PRETRAINED MODELS
    # pretrained_pacmap_dir = None # '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_train45/pacmap/1SecondWindow_1SecondStride/all_pats'  
    pretrained_pacmap_dir = None # '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_val13/pacmap/64SecondWindow_64SecondStride/all_pats'  
    pacmap_basename = 'PaCMAP' # For loading the pretrained model + ANN tree 

    pretrained_phate_dir = None #'/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_train45/phate/1SecondWindow_1SecondStride/all_pats' 

    # Peri-ictal coloring settings
    plot_preictal_color_sec = 60*60*4  # 60*60*4  
    plot_postictal_color_sec = 60*30 

    # ---- HDBSCAN Settings ----
    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100

    FS = 512 # Currently hardcoded in many places

    # Master formatted timestamp file - "All Time Data (ATD)"
    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv'
    
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

    # Get metadata from preloaded pickle
    elif accumulated_data_pickle != None: 
        s = accumulated_data_pickle.split("/")[-1].split("_")
        rewin_windowseconds = int(s[1].replace("secWindow",""))
        rewin_strideseconds = int(s[2].replace("secStride.pkl",""))

    # Gather raw data
    else: 
        file_windowseconds = 64 # Used for directory construction as a string, so must match directory name exactly
        file_strideseconds = 16
        source_dir = f'{parent_dir}/latent_files/{file_windowseconds}SecondWindow_{file_strideseconds}SecondStride'

        # Rewindowing data (Must be multiples of original file duration & stride)
        rewin_windowseconds = 64
        rewin_strideseconds = 16

        subsample_file_factor = 16 # Not re-windowing, subsampled on a whole file level
        subsample_observation_factor = 1 # This will subsample randomly at individual datapoint level (done within the manifold subfunctions after file concatenation)
    

    # ---- PaCMAP Settings ----
    '''
        # NN local, MN: global, FP global and local
        # Phase 1:
        # w_MN = (1 - itr/phase_1_iters) * w_MN_init + itr/phase_1_iters * 3.0     Starts high, goes down to 3
        # w_neighbors = 2.0
        # w_FP = 1.0
        # Phase 2:
        # w_MN = 3.0
        # w_neighbors = 3
        # w_FP = 1
        # Phase 3:
        # w_MN = 0.0
        # w_neighbors = 1.
        # w_FP = 1.
    '''
    apply_pca = True # Before PaCMAP
    pacmap_LR = 0.1 # 0.1 #0.05
    pacmap_NumIters = (500,500,500) # (1500,1500,1500)
    pacmap_NN = None
    pacmap_MN_ratio =  11 # 7 #0.5
    pacmap_FP_ratio = 22 # 11 #2.0


    # ---- PHATE Settings ----
    # Currently hardcoded in sybfunction call
    # knn=15, # Default 5
    # decay=40,  # Default 40
    # phate_metric='euclidean', # 'cosine',
    # phate_solver= 'sgd', # 'smacof',
    # t='auto', 
    # gamma=1,


    # Plotting variables
    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code


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
        ww_means_sentinel = latent_data_fromfile['windowed_means']
        ww_logvars_sentinel = latent_data_fromfile['windowed_logvars']
        ww_z_sentinel = latent_data_fromfile['windowed_z']
        ww_means_allfiles = np.zeros([len(data_filepaths), ww_means_sentinel.shape[0], ww_means_sentinel.shape[1]], dtype=np.float32)
        ww_logvars_allfiles = np.zeros([len(data_filepaths), ww_logvars_sentinel.shape[0], ww_logvars_sentinel.shape[1]], dtype=np.float32)
        ww_z_allfiles = np.zeros([len(data_filepaths), ww_z_sentinel.shape[0], ww_z_sentinel.shape[1]], dtype=np.float32)
        print("Loading all BUILD latent data from files")
        latent_data_fromfile = [''] * len(data_filepaths) # Load all of the data into system RAM - list of [window, latent_dim]
        for i in range(len(data_filepaths)):
            sys.stdout.write(f"\rLoading Pickles: {i}/{len(data_filepaths)}        ") 
            sys.stdout.flush() 
            try:
                with open(data_filepaths[i], "rb") as f: latent_data_fromfile = pickle.load(f)
                ww_means_allfiles[i, :, :] = latent_data_fromfile['windowed_means']
                ww_logvars_allfiles[i, :, :] = latent_data_fromfile['windowed_logvars']
                ww_z_allfiles[i, :, :] = latent_data_fromfile['windowed_z']
            except:
                print(f"Error loading {data_filepaths[i]}")

        # Rewindow the data if needed
        print(f"Original shape of ww_means: {ww_means_allfiles.shape}")
        print(f"Original shape of ww_logvars: {ww_logvars_allfiles.shape}")
        print(f"Original shape of w_mogpreds: {ww_z_allfiles.shape}")
        if (file_windowseconds != rewin_windowseconds) or (file_strideseconds != rewin_strideseconds):
            if (file_windowseconds > rewin_windowseconds) or (file_strideseconds > rewin_strideseconds):
                raise Exception("ERROR: desired window duration or stride is smaller than raw file duration or stride")
            else:
                rewin_means_allfiles, rewin_logvars_allfiles, rewin_z_allfiles = utils_functions.rewindow_data_filewise(
                    ww_means_allfiles, ww_logvars_allfiles, ww_z_allfiles, file_windowseconds, file_strideseconds, rewin_windowseconds, rewin_strideseconds)
                print(f"Rewindowed shape of rewin_means: {rewin_means_allfiles.shape}")
                print(f"Rewindowed shape of rewin_logvars: {rewin_logvars_allfiles.shape}")
                print(f"Rewindowed shape of rewin_mogpreds: {rewin_z_allfiles.shape}")
        else:
            rewin_means_allfiles = ww_means_allfiles
            rewin_logvars_allfiles = ww_logvars_allfiles
            rewin_z_allfiles = ww_z_allfiles
            print("Data NOT re-windowed")

        # # Save the gathered data to save time for re-plotting
        # if (subset_override_dir == None) and (accumulated_data_pickle == None):
        #     savepath = f"{pacmap_savedir}/allDataGathered_subsampleFileFactor{subsample_file_factor}_{rewin_windowseconds}secWindow_{rewin_strideseconds}secStride.pkl"
        #     output_obj = open(savepath, 'wb')
        #     save_dict = {
        #         'rewin_means_allfiles': rewin_means_allfiles,
        #         'rewin_logvars_allfiles': rewin_logvars_allfiles,
        #         'rewin_z_allfiles': rewin_z_allfiles,
        #         'build_start_datetimes': build_start_datetimes,
        #         'build_stop_datetimes': build_stop_datetimes,
        #         'build_pat_ids_list': build_pat_ids_list}
        #     pickle.dump(save_dict, output_obj)
        #     output_obj.close()
        #     print(f"Gathered data saved to one big pickle: {savepath}")

    else: # Get metadata from preloaded pickle
        print("Preloaded single data pickle supplied for all data")
        with open(accumulated_data_pickle, "rb") as f: data_loaded = pickle.load(f)
        rewin_means_allfiles = data_loaded['rewin_means_allfiles']
        rewin_logvars_allfiles = data_loaded['rewin_logvars_allfiles']
        rewin_z_allfiles = data_loaded['rewin_z_allfiles']
        build_start_datetimes = data_loaded['build_start_datetimes']
        build_stop_datetimes = data_loaded['build_stop_datetimes']
        build_pat_ids_list = data_loaded['build_pat_ids_list']

    ### HISTOGRAMS ###
    if run_histos:

        # Create paths and directories for saving dim reduction models and outputs
        histo_dir = f"{parent_dir}/histograms/{rewin_windowseconds}SecondWindow_{rewin_strideseconds}SecondStride"
        if not os.path.exists(histo_dir): os.makedirs(histo_dir)
        if subset_override_dir != None: histo_savedir = f"{histo_dir}/file_subset"
        elif single_pats == []: histo_savedir = f"{histo_dir}/all_pats"
        else: histo_savedir = f"{histo_dir}/{'_'.join(single_pats)}"
        if not os.path.exists(histo_savedir): os.makedirs(histo_savedir)

        raise Exception("TODO: must do custom")

    ### PACMAP ###
    if run_pacmap:

        if pretrained_pacmap_dir == None: 
            reducer = []
            hdb = []
            xy_lims = []
        else: reducer, hdb, xy_lims, _ = manifold_utilities.load_pacmap_objects(pretrained_pacmap_dir, pacmap_basename)

        # Create paths and directories for saving dim reduction models and outputs
        pacmap_dir = f"{parent_dir}/pacmap/{rewin_windowseconds}SecondWindow_{rewin_strideseconds}SecondStride"
        if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
        if subset_override_dir != None: pacmap_savedir = f"{pacmap_dir}/file_subset"
        elif single_pats == []: pacmap_savedir = f"{pacmap_dir}/all_pats"
        else: pacmap_savedir = f"{pacmap_dir}/{'_'.join(single_pats)}"
        if not os.path.exists(pacmap_savedir): os.makedirs(pacmap_savedir)

        # Call the subfunction to create/use pacmap and plot
        axis_20, reducer, hdb, xy_lims = manifold_utilities.pacmap_subfunction(
            premade_PaCMAP = reducer,
            premade_HDBSCAN = hdb,
            xy_lims = xy_lims,
            atd_file = atd_file,
            pat_ids_list=build_pat_ids_list,
            subsample_observation_factor=subsample_observation_factor,
            latent_data_windowed=rewin_means_allfiles, 
            start_datetimes_epoch=build_start_datetimes,  
            stop_datetimes_epoch=build_stop_datetimes,
            win_sec=rewin_windowseconds, 
            stride_sec=rewin_strideseconds, 
            savedir=pacmap_savedir,
            FS = FS,
            apply_pca=apply_pca,
            pacmap_LR = pacmap_LR,
            pacmap_NumIters = pacmap_NumIters,
            pacmap_NN = pacmap_NN,
            pacmap_MN_ratio = pacmap_MN_ratio,
            pacmap_FP_ratio = pacmap_FP_ratio,
            HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
            HDBSCAN_min_samples = HDBSCAN_min_samples,
            plot_preictal_color_sec = plot_preictal_color_sec,
            plot_postictal_color_sec = plot_postictal_color_sec,
            **kwargs)

        # SAVE PACMAP 
        if pretrained_pacmap_dir == None:
            manifold_utilities.save_pacmap_objects(
                pacmap_dir=pacmap_savedir,
                axis=axis_20,
                reducer=reducer, 
                hdb=hdb, 
                xy_lims=xy_lims)


    ### PHATE ###
    if run_phate:

        if pretrained_phate_dir == None: 
            reducer = []
            hdb = []
            xy_lims = []
        else: reducer, hdb, xy_lims, _ = manifold_utilities.load_phate_objects(pretrained_phate_dir)

        # Create paths and directories for saving dim reduction models and outputs
        phate_dir = f"{parent_dir}/phate/{rewin_windowseconds}SecondWindow_{rewin_strideseconds}SecondStride"
        if not os.path.exists(phate_dir): os.makedirs(phate_dir)
        if subset_override_dir != None: phate_savedir = f"{phate_dir}/file_subset"
        elif single_pats == []: phate_savedir = f"{phate_dir}/all_pats"
        else: phate_savedir = f"{phate_dir}/{'_'.join(single_pats)}"
        if not os.path.exists(phate_savedir): os.makedirs(phate_savedir)
        ax01, reducer, hdb, xy_lims = manifold_utilities.phate_subfunction(  
            premade_PHATE = reducer,
            premade_HDBSCAN = hdb,
            xy_lims = xy_lims,
            atd_file=atd_file,
            pat_ids_list=build_pat_ids_list,
            subsample_observation_factor=subsample_observation_factor,
            latent_data_windowed=rewin_means_allfiles, 
            start_datetimes_epoch=build_start_datetimes,  
            stop_datetimes_epoch=build_stop_datetimes,
            epoch=0, 
            FS=FS, 
            win_sec=rewin_windowseconds, 
            stride_sec=rewin_strideseconds, 
            savedir=phate_savedir,
            HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
            HDBSCAN_min_samples = HDBSCAN_min_samples,
            plot_preictal_color_sec = plot_preictal_color_sec,
            plot_postictal_color_sec = plot_postictal_color_sec,
            **kwargs)
        
        # SAVE PHATE 
        if pretrained_phate_dir == None:
            manifold_utilities.save_phate_objects(
                phate_dir=phate_savedir, 
                axis=ax01, 
                reducer=reducer, 
                hdb=hdb, 
                xy_lims=xy_lims)



    # if run_histo:
    #     ### HISTOGRAM LATENT ###
    #     # Generation data
    #     print("Histogram on generation data")
    #     if single_pats == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    #     else: histo_dir = f"{model_dir}/histo_latent/{'_'.join(single_pats)}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    #     if not os.path.exists(histo_dir): os.makedirs(histo_dir)
    #     manifold_utilities.histogram_latent(
    #         pat_ids_list=build_pat_ids_list,
    #         latent_data_windowed=latent_data_windowed_generation, 
    #         start_datetimes_epoch=build_start_datetimes,  
    #         stop_datetimes_epoch=build_stop_datetimes,
    #         epoch=epoch, 
    #         win_sec=win_sec, 
    #         stride_sec=stride_sec, 
    #         savedir=histo_dir,
    #         FS = FS)

    #     # Eval data
    #     if eval_filepaths != []:
    #         print("Histogram on evaluation data")
    #         if single_pats == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
    #         else: histo_dir = f"{model_dir}/histo_latent/{'_'.join(single_pats)}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
    #         if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    #         manifold_utilities.histogram_latent(
    #             pat_ids_list=eval_pat_ids_list,
    #             latent_data_windowed=latent_data_windowed_eval, 
    #             start_datetimes_epoch=eval_start_datetimes,  
    #             stop_datetimes_epoch=eval_stop_datetimes,
    #             epoch=epoch, 
    #             win_sec=win_sec, 
    #             stride_sec=stride_sec, 
    #             savedir=histo_dir,
    #             FS = FS)



