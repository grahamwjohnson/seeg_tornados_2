'''
Ad-hoc script to run PaCMAP/PHATE/Histograms on latent files
'''

import os, glob
import pickle
from  utilities import utils_functions

if __name__ == "__main__":

    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code

    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/pangolin_ripple/trained_models/all_time_data_01092023_112957.csv'

    # Source data selection
    model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025'
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/jackal'
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/pangolin_ripple/trained_models/pangolin_spat113_finetune'
    # pat_ids_list = ['Epat34']
    single_pat = [] #'Epat35'  # if [] will do all pats
    epoch = 141 # 39 # 141 , 999 to debug
    latent_subdir = f'latent_files/Epoch{epoch}'
    win_sec = 60 # 60, 1.0
    stride_sec = 30 # 30, 1.0 

    # pacmap_build_strs = ['train', 'valfinetune']
    # pacmap_eval_strs = ['valunseen']
    pacmap_build_strs = ['train']
    pacmap_eval_strs = ['valfinetune', 'valunseen'] # Typically will use valfinetune in build...?
    
    FS = 512 

    # PaCMAP Settings
    apply_pca = True # Before PaCMAP
    pca_comp = 100
    pacmap_MedDim_numdims = 10
    pacmap_LR = 1 #0.05
    pacmap_NumIters = (1000,1000,1000)
    pacmap_NN = None
    pacmap_MN_ratio = 7 #0.5
    pacmap_FP_ratio = 11 #2.0

    # PHATE Settings
    precomputed_nn_path = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025/phate/Epoch141/60SecondWindow_30SecondStride/all_pats/phate_gen/nn_pickles/Window60_Stride30_epoch141_angular_knn5_KNN_INDICES.pkl' # []
    precomputed_dist_path = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025/phate/Epoch141/60SecondWindow_30SecondStride/all_pats/phate_gen/nn_pickles/Window60_Stride30_epoch141_angular_knn5_KNN_DISTANCES.pkl' #[]
    precomputed_nn = [] # dummy
    precomputed_dist = [] # dummy
    phate_annoy_tree_size = 20
    phate_knn = 5
    phate_decay = 15
    phate_metric = 'angular' # 'angular', 'euclidean' # Used by custom ANNOY function, angular=cosine for ANNOY
    phate_solver = 'smacof'  # 'smacof', 'sgd' # I think SGD uses less RAM because it's stochastic

    # HDBSCAN Settings
    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100

    # Plotting Settings
    plot_preictal_color_sec = 60*60*4
    plot_postictal_color_sec = 60*30 #60*60*4

    # Create paths and create pacmap directory for saving dim reduction models and outputs
    latent_dir = f"{model_dir}/{latent_subdir}/{win_sec}SecondWindow_{stride_sec}SecondStride" 
    pacmap_dir = f"{model_dir}/pacmap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    phate_dir = f"{model_dir}/phate/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    if not os.path.exists(phate_dir): os.makedirs(phate_dir)

    ### GENERATION DATA ###
    build_filepaths = [] # Collect pacmap build files - i.e. what data is being used to construct data manifold approximator
    for i in range(len(pacmap_build_strs)):
        dir_curr = f"{latent_dir}/{pacmap_build_strs[i]}"
        if single_pat == []: build_filepaths = build_filepaths + glob.glob(dir_curr + f'/*.pkl')
        else: build_filepaths = build_filepaths + glob.glob(dir_curr + f'/{single_pat}*.pkl')
    assert (build_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (build_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
    build_start_datetimes, build_stop_datetimes = utils_functions.filename_to_datetimes([s.split("/")[-1] for s in build_filepaths]) # Get start/stop datetimes
    build_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in build_filepaths] # Get the build pat_ids
    print("Loading all BUILD latent data from files")
    latent_data_windowed_generation = [''] * len(build_filepaths) # Load all of the data into system RAM - list of [window, latent_dim]
    for i in range(len(build_filepaths)):
        with open(build_filepaths[i], "rb") as f: 
            latent_data_windowed_generation[i] = pickle.load(f)

    ### EVAL DATA ###
    eval_filepaths = [] # these data are not used to build the manifold projections, but are just run through them after construction
    for i in range(len(pacmap_eval_strs)):
        dir_curr = f"{latent_dir}/{pacmap_eval_strs[i]}"
        if single_pat == []: eval_filepaths = eval_filepaths + glob.glob(dir_curr + f'/*.pkl')
        else: eval_filepaths = eval_filepaths + glob.glob(dir_curr + f'/{single_pat}*.pkl')
    if eval_filepaths != []: # May not have any eval data available depending on selections
        assert (eval_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (eval_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
        eval_start_datetimes, eval_stop_datetimes = utils_functions.filename_to_datetimes([s.split("/")[-1] for s in eval_filepaths]) # Get start/stop datetimes
        eval_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in eval_filepaths] # Get the eval pat_ids
        print("Loading all EVAL latent data from files")
        latent_data_windowed_eval = [''] * len(eval_filepaths)  # Load all of the data into system RAM - list of [window, latent_dim]
        for i in range(len(eval_filepaths)):
            with open(eval_filepaths[i], "rb") as f: 
                latent_data_windowed_eval[i] = pickle.load(f)
    else: print(f"WARNING: no evaluation data for {single_pat} in these categories: {pacmap_eval_strs}")


    ### PACMAP GENERATION ###
         
    # # Call the subfunction to create/use pacmap and plot
    # if single_pat == []: pacmap_savedir = f"{pacmap_dir}/all_pats/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_generation"
    # else: pacmap_savedir = f"{pacmap_dir}/{single_pat}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_generation"
    # axis_20, reducer, reducer_MedDim, hdb, pca, xy_lims, xy_lims_PCA, xy_lims_RAW_DIMS = utils_functions.pacmap_subfunction(
    #     atd_file = atd_file,
    #     pat_ids_list=build_pat_ids_list,
    #     latent_data_windowed=latent_data_windowed_generation, 
    #     start_datetimes_epoch=build_start_datetimes,  
    #     stop_datetimes_epoch=build_stop_datetimes,
    #     epoch=epoch, 
    #     win_sec=win_sec, 
    #     stride_sec=stride_sec, 
    #     savedir=pacmap_savedir,
    #     FS = FS,
    #     apply_pca=apply_pca,
    #     pca_comp=pca_comp,
    #     pacmap_MedDim_numdims = pacmap_MedDim_numdims,
    #     pacmap_LR = pacmap_LR,
    #     pacmap_NumIters = pacmap_NumIters,
    #     pacmap_NN = pacmap_NN,
    #     pacmap_MN_ratio = pacmap_MN_ratio,
    #     pacmap_FP_ratio = pacmap_FP_ratio,
    #     HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
    #     HDBSCAN_min_samples = HDBSCAN_min_samples,
    #     plot_preictal_color_sec = plot_preictal_color_sec,
    #     plot_postictal_color_sec = plot_postictal_color_sec,
    #     **kwargs)

    # # SAVE OBJECTS
    # utils_functions.save_pacmap_objects(
    #     pacmap_dir=pacmap_savedir,
    #     epoch=epoch,
    #     axis=axis_20,
    #     reducer=reducer, 
    #     reducer_MedDim=reducer_MedDim, 
    #     hdb=hdb, 
    #     pca=pca, 
    #     xy_lims=xy_lims, 
    #     xy_lims_PCA=xy_lims_PCA, 
    #     xy_lims_RAW_DIMS=xy_lims_RAW_DIMS)

    ### PACMAP EVAL ONLY ###
    
    # if eval_filepaths != []:
        # # Call the subfunction to create/use pacmap and plot
        # pacmap_savedir = f"{pacmap_dir}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_eval"
        # utils_functions.pacmap_subfunction(
        #     atd_file=atd_file,
        #     pat_ids_list=eval_pat_ids_list,
        #     latent_data_windowed=latent_data_windowed_eval, 
        #     start_datetimes_epoch=eval_start_datetimes,  
        #     stop_datetimes_epoch=eval_stop_datetimes,
        #     epoch=epoch, 
        #     win_sec=win_sec, 
        #     stride_sec=stride_sec, 
        #     savedir=pacmap_savedir,
        #     FS = FS,
        #     apply_pca=apply_pca,
        #     pca_comp=pca_comp,
        #     pacmap_MedDim_numdims = pacmap_MedDim_numdims,
        #     pacmap_LR = pacmap_LR,
        #     pacmap_NumIters = pacmap_NumIters,
        #     pacmap_NN = pacmap_NN,
        #     pacmap_MN_ratio = pacmap_MN_ratio,
        #     pacmap_FP_ratio = pacmap_FP_ratio,
        #     HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
        #     HDBSCAN_min_samples = HDBSCAN_min_samples,
        #     plot_preictal_color_sec = plot_preictal_color_sec,
        #     plot_postictal_color_sec = plot_postictal_color_sec,
        #     xy_lims = xy_lims,
        #     xy_lims_RAW_DIMS = xy_lims_RAW_DIMS,
        #     xy_lims_PCA = xy_lims_PCA,
        #     premade_PaCMAP = reducer,
        #     premade_PaCMAP_MedDim = reducer_MedDim,
        #     premade_PCA = pca,
        #     premade_HDBSCAN = hdb,
        #     **kwargs)


    ### PHATE GENERATION ###
    if single_pat == []: phate_savedir = f"{phate_dir}/all_pats/phate_gen"
    else: phate_savedir = f"{phate_dir}/{single_pat}/phate_gen"

    # Pull in precomputed ANNOY values if any
    if (precomputed_nn_path != []) and (precomputed_dist_path != []):
        with open(precomputed_nn_path, "rb") as f: precomputed_nn = pickle.load(f)
        with open(precomputed_dist_path, "rb") as f: precomputed_dist = pickle.load(f)

    phate_ax20, phate, phate_hdb, phate_xy_lims = utils_functions.phate_subfunction(
        atd_file = atd_file,
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=latent_data_windowed_generation, 
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        epoch=epoch, 
        win_sec=win_sec, 
        stride_sec=stride_sec, 
        savedir=phate_savedir,
        FS = FS,
        HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
        HDBSCAN_min_samples = HDBSCAN_min_samples,
        plot_preictal_color_sec = plot_preictal_color_sec,
        plot_postictal_color_sec = plot_postictal_color_sec,  
        interictal_contour=False,
        knn=phate_knn,
        decay=phate_decay,
        phate_metric=phate_metric,
        phate_solver=phate_solver,
        verbose=True,
        xy_lims = [],
        phate_annoy_tree_size = phate_annoy_tree_size,
        knn_indices = precomputed_nn,
        knn_distances = precomputed_dist,
        premade_PHATE = [],
        premade_HDBSCAN = [], 
        **kwargs)


    ### PAHTE SAVE OBJECTS ###
    # utils_functions.save_phate_objects(
    #     savedir=phate_savedir,
    #     epoch=epoch,
    #     axis=phate_ax20,
    #     phate=phate, 
    #     hdb=hdb, 
    #     xy_lims=xy_lims)

    ### PHATE EVAL ###
    # TODO




    ### HISTOGRAM LATENT ###

    # # Generation data
    # if single_pat == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    # else: histo_dir = f"{model_dir}/histo_latent/{single_pat}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    # if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    # utils_functions.histogram_latent(
    #     pat_ids_list=build_pat_ids_list,
    #     latent_data_windowed=latent_data_windowed_generation, 
    #     start_datetimes_epoch=build_start_datetimes,  
    #     stop_datetimes_epoch=build_stop_datetimes,
    #     epoch=epoch, 
    #     win_sec=win_sec, 
    #     stride_sec=stride_sec, 
    #     savedir=f"{histo_dir}/histo_generation",
    #     FS = FS)

    # # Eval data
    # if single_pat == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
    # else: histo_dir = f"{model_dir}/histo_latent/{single_pat}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
    # if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    # utils_functions.histogram_latent(
    #     pat_ids_list=eval_pat_ids_list,
    #     latent_data_windowed=latent_data_windowed_eval, 
    #     start_datetimes_epoch=eval_start_datetimes,  
    #     stop_datetimes_epoch=eval_stop_datetimes,
    #     epoch=epoch, 
    #     win_sec=win_sec, 
    #     stride_sec=stride_sec, 
    #     savedir=f"{histo_dir}/histo_eval",
    #     FS = FS)



