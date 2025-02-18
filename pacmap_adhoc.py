'''
Ad-hoc script to run PaCMAP on latent files
'''

import os, glob
import pickle
from  utilities import utils_functions

if __name__ == "__main__":

    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code

    # Source data selection
    model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025'
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/jackal'
    # pat_ids_list = ['Epat34']
    single_pat = 'Epat34'
    epoch = 141 # 39 # 141 
    latent_subdir = f'latent_files/Epoch{epoch}'
    win_sec = 1.0 #60 
    stride_sec = 1.0 #30 

    # pacmap_build_strs = ['train', 'valfinetune']
    # pacmap_eval_strs = ['valunseen']
    pacmap_build_strs = ['train']
    pacmap_eval_strs = ['']
    
    FS = 512 
    #TODO : pacmap settings 
    pacmap_MedDim_numdims = 10
    pacmap_LR = 0.05
    pacmap_NumIters = (900,900,900)

    pacmap_NN = None
    pacmap_MN_ratio = 10 #0.5 4 
    pacmap_FP_ratio = 10 #2.0 7

    pacmap_MN_ratio_MedDim = pacmap_MN_ratio
    pacmap_FP_ratio_MedDim = pacmap_FP_ratio

    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100
    plot_preictal_color_sec = 60*60*4
    plot_postictal_color_sec = 60*10 #60*60*4


    # Create paths and create pacmap directory for saving pacmap models and outputs
    latent_dir = f"{model_dir}/{latent_subdir}/{win_sec}SecondWindow_{stride_sec}SecondStride" 
    pacmap_dir = f"{model_dir}/pacmap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/adhoc"
    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)

    ### PACMAP GENERATION ###
 
    # Collect pacmap build files - i.e. what data is being used to construct data manifold approximator
    build_filepaths = []
    for i in range(len(pacmap_build_strs)):
        dir_curr = f"{latent_dir}/{pacmap_build_strs[i]}"
        build_filepaths = build_filepaths + glob.glob(dir_curr + f'/{single_pat}*.pkl')

    # Double check window and stride are correct based on file naming [HARDCODED]
    assert (build_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (build_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow")

    # Get start/stop datetimes
    build_start_datetimes, build_stop_datetimes = utils_functions.filename_to_datetimes([s.split("/")[-1] for s in build_filepaths])

    # Get the build pat_ids
    build_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in build_filepaths]

    # Load all of the data into system RAM - list of [window, latent_dim]
    print("Loading all BUILD latent data from files")
    latent_data_windowed = [''] * len(build_filepaths)
    for i in range(len(build_filepaths)):
        with open(build_filepaths[i], "rb") as f: 
            latent_data_windowed[i] = pickle.load(f)
         
    # # Call the subfunction to create/use pacmap and plot
    # reducer, reducer_MedDim, hdb, pca, xy_lims, xy_lims_PCA, xy_lims_RAW_DIMS = utils_functions.pacmap_subfunction(
    #     pat_ids_list=build_pat_ids_list,
    #     latent_data_windowed=latent_data_windowed, 
    #     start_datetimes_epoch=build_start_datetimes,  
    #     stop_datetimes_epoch=build_stop_datetimes,
    #     epoch=epoch, 
    #     win_sec=win_sec, 
    #     stride_sec=stride_sec, 
    #     savedir=f"{pacmap_dir}/{single_pat}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_generation",
    #     FS = FS,
    #     pacmap_MedDim_numdims = pacmap_MedDim_numdims,
    #     pacmap_LR = pacmap_LR,
    #     pacmap_NumIters = pacmap_NumIters,
    #     pacmap_NN = pacmap_NN,
    #     pacmap_MN_ratio = pacmap_MN_ratio,
    #     pacmap_FP_ratio = pacmap_FP_ratio,
    #     pacmap_MN_ratio_MedDim = pacmap_MN_ratio_MedDim,
    #     pacmap_FP_ratio_MedDim = pacmap_FP_ratio_MedDim,
    #     HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
    #     HDBSCAN_min_samples = HDBSCAN_min_samples,
    #     plot_preictal_color_sec = plot_preictal_color_sec,
    #     plot_postictal_color_sec = plot_postictal_color_sec,
    #     **kwargs)


    ### HISTOGRAM LATENT ###

    # Generation data
    histo_dir = f"{model_dir}/histo_latent/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    utils_functions.histogram_latent(
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=latent_data_windowed, 
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        epoch=epoch, 
        win_sec=win_sec, 
        stride_sec=stride_sec, 
        savedir=f"{histo_dir}/{single_pat}/histo_generation",
        FS = FS)










    # ### PACMAP EVAL ONLY ###

    # # Now run the desired data through the pacmap/hdbscan models
    # # Collect pacmap eval files - i.e. what data is being used to construct data manifold approximator
    # eval_filepaths = []
    # for i in range(len(pacmap_eval_strs)):
    #     dir_curr = f"{latent_dir}/{pacmap_eval_strs[i]}"
    #     eval_filepaths = eval_filepaths + glob.glob(dir_curr + '/*.pkl')

    # # Double check window and stride are correct based on file naming [HARDCODED]
    # assert (eval_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (eval_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow")

    # # Get start/stop datetimes
    # eval_start_datetimes, eval_stop_datetimes = utils_functions.filename_to_datetimes([s.split("/")[-1] for s in eval_filepaths])

    # # Get the eval pat_ids
    # eval_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in eval_filepaths]

    # # Load all of the data into system RAM - list of [window, latent_dim]
    # print("Loading all eval latent data from files")
    # latent_data_windowed = [''] * len(eval_filepaths)
    # for i in range(len(eval_filepaths)):
    #     with open(eval_filepaths[i], "rb") as f: 
    #         latent_data_windowed[i] = pickle.load(f)
         
    # # Call the subfunction to create/use pacmap and plot
    # pacmap_subfunction(
    #     pat_ids_list=eval_pat_ids_list,
    #     latent_data_windowed=latent_data_windowed, 
    #     start_datetimes_epoch=eval_start_datetimes,  
    #     stop_datetimes_epoch=eval_stop_datetimes,
    #     epoch=epoch, 
    #     win_sec=win_sec, 
    #     stride_sec=stride_sec, 
    #     savedir=f"{pacmap_dir}/pacmap_eval_only",
    #     xy_lims = xy_lims,
    #     xy_lims_RAW_DIMS = xy_lims_RAW_DIMS,
    #     xy_lims_PCA = xy_lims_PCA,
    #     premade_PaCMAP = reducer,
    #     premade_PaCMAP_MedDim = reducer_MedDim,
    #     premade_PCA = pca,
    #     premade_HDBSCAN = hdb,
    #     **kwargs)

    # # SAVE OBJECTS
    # save_pacmap_objects(
    #     pacmap_dir=pacmap_dir,
    #     epoch=epoch,
    #     reducer=reducer, 
    #     reducer_MedDim=reducer_MedDim, 
    #     hdb=hdb, 
    #     pca=pca, 
    #     xy_lims=xy_lims, 
    #     xy_lims_PCA=xy_lims_PCA, 
    #     xy_lims_RAW_DIMS=xy_lims_RAW_DIMS)