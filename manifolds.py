import os, glob
import pickle
from  utilities import utils_functions
import numpy as np

'''
@author: grahamwjohnson
March 2025

Ad-hoc script to run UMAP/PaCMAP/PHATE/Histograms on latent files
'''

if __name__ == "__main__":

    # Master formatted timestamp file - "All Time Data (ATD)"
    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/pangolin_ripple/trained_models/all_time_data_01092023_112957.csv'

    # Source data selection
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025'
    model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats/trained_models/dataset_train90.0_val10.0/gar_Tue_Feb_18_13_39_07_2025'
    single_pat = [] # 'Spat18' # [] #'Epat35'  # if [] will do all pats  # TODO: modify to take a selection of patients
    epoch = 165 # 39 # 141 , 999 to debug
    latent_subdir = f'latent_files/Epoch{epoch}'
    win_sec = 60 # 60, 10  # Must match strings in directory name exactly (e.g. 1.0 not 1)
    stride_sec = 30 # 30, 10 

    # build_strs = ['train', 'valfinetune']
    # eval_strs = ['valunseen']
    build_strs = ['train']  # build selections will be use to train/construct/fit the manifold models
    eval_strs = ['valfinetune', 'valunseen'] # Typically will use valfinetune in build...?
    
    FS = 512 # Currently hardcoded in many places

    # HDBSCAN Settings
    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100

    # Plotting Settings
    plot_preictal_color_sec = 60*60*4
    plot_postictal_color_sec = 60*30 #60*60*4

    # UMAP settings
    umap_output_metric = 'euclidean' # 'euclidean', 'hyperboloid'
    umap_n_neighbors = 15 # 15 default
    umap_metric = 'cosine'  # euclidean, cosine
    umap_init = 'random'  # 'spectral'   # default spectral
    umap_min_dist = 0.1 # 0-1, 0.1 default
    umap_densmap = False  # allows for local densities
    umap_dens_lambda = 0.1 # default 2.0, higher means more local density
    umap_spread = 1 # 0.5-2, default 1
    umap_local_connectivity = 1 # 1 default, can go higher
    umap_repulsion_strength = 1 # 1 default, can go higher

    # PaCMAP Settings
    # TODO take in previously calculated NN
    exclude_self_pat = False
    apply_pca = True # Before PaCMAP
    pca_comp = 100
    pacmap_MedDim_numdims = 10
    pacmap_LR = 0.1 #0.05
    pacmap_NumIters = (500,500,500)
    pacmap_NN = None
    pacmap_MN_ratio = 7 # 7 #0.5
    pacmap_FP_ratio = 11 # 11 #2.0

    # PHATE Settings
    custom_nn_bool = False
    precomputed_nn_path = [] #'/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025/phate/Epoch141/60SecondWindow_30SecondStride/all_pats/phate_gen/nn_pickles/Window60_Stride30_epoch141_angular_knn5_KNN_INDICES.pkl' # []
    precomputed_dist_path = [] #'/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025/phate/Epoch141/60SecondWindow_30SecondStride/all_pats/phate_gen/nn_pickles/Window60_Stride30_epoch141_angular_knn5_KNN_DISTANCES.pkl' #[]
    precomputed_nn = [] # dummy
    precomputed_dist = [] # dummy
    phate_annoy_tree_size = 20 # For NN approximations
    phate_knn = 5
    phate_decay = 40 # Bigger means more local diffusion structure
    phate_metric = 'angular' # 'angular', 'euclidean' # Used by custom ANNOY function, angular=cosine for ANNOY
    phate_solver = 'smacof'  # 'smacof' (longer), 'sgd' 
    rand_subset_pat_bool = False # False plots all pats in their own row of plots, if more than ~5 pats, should probably set to True
    num_rand_pats_plot = 4 # Only applicable if 'rand_subset_pat_bool' is True

    # Plotting variables
    kwargs = {}
    kwargs['seiz_type_list'] = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'] # Leftward overwites rightward
    kwargs['seiz_plot_mult'] = [1,       3,     5,              7,    9,                           11,        13,            15] # Assuming increasing order, NOTE: base value of 3 is added in the code

    # Create paths and create pacmap directory for saving dim reduction models and outputs
    umap_dir = f"{model_dir}/umap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    latent_dir = f"{model_dir}/{latent_subdir}/{win_sec}SecondWindow_{stride_sec}SecondStride" 
    pacmap_dir = f"{model_dir}/pacmap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    phate_dir = f"{model_dir}/phate/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    umap_dir = f"{model_dir}/umap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    if not os.path.exists(phate_dir): os.makedirs(phate_dir)
    if not os.path.exists(umap_dir): os.makedirs(umap_dir)

    ### GENERATION DATA ###
    build_filepaths = [] # Collect pacmap build files - i.e. what data is being used to construct data manifold approximator
    for i in range(len(build_strs)):
        dir_curr = f"{latent_dir}/{build_strs[i]}"
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
    for i in range(len(eval_strs)):
        dir_curr = f"{latent_dir}/{eval_strs[i]}"
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
    else: print(f"WARNING: no evaluation data for {single_pat} in these categories: {eval_strs}")


    ### UMAP GENERATION ###
    # Call the subfunction to create/use umap and plot
    if single_pat == []: umap_savedir = f"{umap_dir}/all_pats/umap_generation"
    else: umap_savedir = f"{umap_dir}/{single_pat}/umap_generation"
    axis_00, reducer, hdb, xy_lims = utils_functions.umap_subfunction(
        atd_file = atd_file,
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=latent_data_windowed_generation, 
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        epoch=epoch, 
        win_sec=win_sec, 
        stride_sec=stride_sec, 
        savedir=umap_savedir,
        FS = FS,
        apply_pca=apply_pca,
        pca_comp=pca_comp,
        output_metric=umap_output_metric,
        n_neighbors=umap_n_neighbors,
        metric=umap_metric,
        min_dist=umap_min_dist,
        densmap=umap_densmap,
        dens_lambda=umap_dens_lambda,
        init=umap_init,
        spread=umap_spread,
        local_connectivity=umap_local_connectivity, 
        repulsion_strength=umap_repulsion_strength,
        HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
        HDBSCAN_min_samples = HDBSCAN_min_samples,
        plot_preictal_color_sec = plot_preictal_color_sec,
        plot_postictal_color_sec = plot_postictal_color_sec,
        **kwargs)


    ### PACMAP GENERATION ###
    # Call the subfunction to create/use pacmap and plot
    if single_pat == []: pacmap_savedir = f"{pacmap_dir}/all_pats/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_generation"
    else: pacmap_savedir = f"{pacmap_dir}/{single_pat}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_generation"
    axis_20, reducer, reducer_MedDim, hdb, pca, xy_lims, xy_lims_PCA, xy_lims_RAW_DIMS = utils_functions.pacmap_subfunction(
        atd_file = atd_file,
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=latent_data_windowed_generation, 
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        epoch=epoch, 
        win_sec=win_sec, 
        stride_sec=stride_sec, 
        savedir=pacmap_savedir,
        FS = FS,
        apply_pca=apply_pca,
        pca_comp=pca_comp,
        exclude_self_pat=exclude_self_pat,
        pacmap_MedDim_numdims = pacmap_MedDim_numdims,
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

    # SAVE OBJECTS
    utils_functions.save_pacmap_objects(
        pacmap_dir=pacmap_savedir,
        epoch=epoch,
        axis=axis_20,
        reducer=reducer, 
        reducer_MedDim=reducer_MedDim, 
        hdb=hdb, 
        pca=pca, 
        xy_lims=xy_lims, 
        xy_lims_PCA=xy_lims_PCA, 
        xy_lims_RAW_DIMS=xy_lims_RAW_DIMS)

    ## PACMAP EVAL ONLY ###
    if eval_filepaths != []:
        # Call the subfunction to create/use pacmap and plot
        if single_pat == []: pacmap_savedir = f"{pacmap_dir}/all_pats/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_eval"
        else: pacmap_savedir = f"{pacmap_dir}/{single_pat}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_eval"
        utils_functions.pacmap_subfunction(
            atd_file=atd_file,
            pat_ids_list=eval_pat_ids_list,
            latent_data_windowed=latent_data_windowed_eval, 
            start_datetimes_epoch=eval_start_datetimes,  
            stop_datetimes_epoch=eval_stop_datetimes,
            epoch=epoch, 
            win_sec=win_sec, 
            stride_sec=stride_sec, 
            savedir=pacmap_savedir,
            FS = FS,
            apply_pca=apply_pca,
            pca_comp=pca_comp,
            exclude_self_pat=exclude_self_pat,
            pacmap_MedDim_numdims = pacmap_MedDim_numdims,
            pacmap_LR = pacmap_LR,
            pacmap_NumIters = pacmap_NumIters,
            pacmap_NN = pacmap_NN,
            pacmap_MN_ratio = pacmap_MN_ratio,
            pacmap_FP_ratio = pacmap_FP_ratio,
            HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
            HDBSCAN_min_samples = HDBSCAN_min_samples,
            plot_preictal_color_sec = plot_preictal_color_sec,
            plot_postictal_color_sec = plot_postictal_color_sec,
            xy_lims = xy_lims,
            xy_lims_RAW_DIMS = xy_lims_RAW_DIMS,
            xy_lims_PCA = xy_lims_PCA,
            premade_PaCMAP = reducer,
            premade_PaCMAP_MedDim = reducer_MedDim,
            premade_PCA = pca,
            premade_HDBSCAN = hdb,
            **kwargs)


    ### PHATE GENERATION ###
    if single_pat == []: phate_savedir = f"{phate_dir}/all_pats/phate_gen"
    else: phate_savedir = f"{phate_dir}/{single_pat}/phate_gen"

    # Pull in precomputed ANNOY values if any
    if (precomputed_nn_path != []) and (precomputed_dist_path != []):
        with open(precomputed_nn_path, "rb") as f: precomputed_nn = pickle.load(f)
        with open(precomputed_dist_path, "rb") as f: precomputed_dist = pickle.load(f)

    # Generate random idxs to plot
    unique_pats = list(set(build_pat_ids_list))
    if rand_subset_pat_bool:
        np.random.seed(seed=None) 
        plot_pat_ids = [unique_pats[np.random.randint(0, len(unique_pats))] for i in range(num_rand_pats_plot)]
    else:
        plot_pat_ids = unique_pats

    # Run the PHATE subfunction on generate data - plots are made/saved within this function
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
        custom_nn_bool = custom_nn_bool,
        phate_annoy_tree_size = phate_annoy_tree_size,
        knn_indices = precomputed_nn,
        knn_distances = precomputed_dist,
        premade_PHATE = [],
        premade_HDBSCAN = [], 
        plot_pat_ids = plot_pat_ids,
        **kwargs)


    ### PAHTE SAVE OBJECTS ### TODO code up
    # utils_functions.save_phate_objects(
    #     savedir=phate_savedir,
    #     epoch=epoch,
    #     axis=phate_ax20,
    #     phate=phate, 
    #     hdb=hdb, 
    #     xy_lims=xy_lims)

    ### PHATE EVAL ###
    # TODO
    # if eval_filepaths != []:


    ### HISTOGRAM LATENT ###
    
    # Generation data
    print("Histogram on generation data")
    if single_pat == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    else: histo_dir = f"{model_dir}/histo_latent/{single_pat}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    utils_functions.histogram_latent(
        pat_ids_list=build_pat_ids_list,
        latent_data_windowed=latent_data_windowed_generation, 
        start_datetimes_epoch=build_start_datetimes,  
        stop_datetimes_epoch=build_stop_datetimes,
        epoch=epoch, 
        win_sec=win_sec, 
        stride_sec=stride_sec, 
        savedir=histo_dir,
        FS = FS)

    # Eval data
    if eval_filepaths != []:
        print("Histogram on evaluation data")
        if single_pat == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
        else: histo_dir = f"{model_dir}/histo_latent/{single_pat}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
        if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
        utils_functions.histogram_latent(
            pat_ids_list=eval_pat_ids_list,
            latent_data_windowed=latent_data_windowed_eval, 
            start_datetimes_epoch=eval_start_datetimes,  
            stop_datetimes_epoch=eval_stop_datetimes,
            epoch=epoch, 
            win_sec=win_sec, 
            stride_sec=stride_sec, 
            savedir=histo_dir,
            FS = FS)



