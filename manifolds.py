import os, glob
import pickle
from  utilities import manifold_utilities
import numpy as np
import random

'''
@author: grahamwjohnson
March 2025

Ad-hoc script to run UMAP/PR-PaCMAP/PaCMAP/PHATE/Histograms on latent files
'''

def custom_paramrep_weight_schedule(epoch: int):
    """Weight schedule for ParamRepulsor."""
    # Dynamic
    if epoch < 300: # Default 200
        w_nn = 4.0 # Default 4
        w_fp = 8.0 # Default 8
        w_mn = 0 # Default 0

    # elif epoch < 500: # this pahse does not exist in default
    #     w_nn = 8.0 
    #     w_fp = 2.0 
    #     w_mn = -1 

    else:
        w_nn = 1.0 # Default 1
        w_fp = 8.0 # Default 8
        w_mn = -12.0 # Default -12

    # # Static
    # w_nn = 1.0
    # w_fp = 10.0
    # w_mn = 1.0

    weight = np.array([w_nn, w_fp, w_mn])
    return weight

if __name__ == "__main__":

    run_prpacmap = False
    run_umap = False
    run_pacmap = False
    run_phate = False
    run_kohenen = True
    run_histo = False

    # Master formatted timestamp file - "All Time Data (ATD)"
    atd_file = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/pangolin_ripple/trained_models/all_time_data_01092023_112957.csv'

    # Source data selection
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/pangolin_Thu_Jan_30_18_29_14_2025'
    model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats/trained_models/dataset_train90.0_val10.0/tmp_incatern'
    # model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/Mobo_pats/trained_models/dataset_train90.0_val10.0/tmp_testing'
    single_pats = ['Epat27'] # ['Epat27', 'Epat28', 'Epat30', 'Epat31', 'Epat33', 'Epat34', 'Epat35', 'Epat37', 'Epat39', 'Epat41'] # [] # 'Spat18' # 'Spat18' # [] #'Epat35'  # if [] will do all pats  # TODO: modify to take a selection of patients
    epoch = 33 # 39 # 141 , 999 to debug
    latent_subdir = f'latent_files/Epoch{epoch}'
    win_sec = 64 # 60, 10  # Must match strings in directory name exactly (e.g. 1.0 not 1)
    stride_sec = 64 # 30, 10 

    # build_strs = ['train', 'valfinetune']
    # eval_strs = ['valunseen']
    build_strs = ['train']  # build selections will be use to train/construct/fit the manifold models
    eval_strs = ['valfinetune', 'valunseen'] # Typically will use valfinetune in build...?
    
    FS = 512 # Currently hardcoded in many places

    # HDBSCAN Settings
    HDBSCAN_min_cluster_size = 200
    HDBSCAN_min_samples = 100

    # Plotting Settings
    plot_preictal_color_sec = 60*60*1 #60*60*4
    plot_postictal_color_sec = 0 #60*10 #60*60*4

    # Kohenen Settings
    som_batch_size = 256
    som_lr = 0.5
    som_epochs = 10
    som_gridsize = 20
    som_lr_epoch_decay = 0.90
    som_sigma = int(som_gridsize/2)
    som_sigma_epoch_decay = 0.90

    # ParamRepulsor Settings (unless random override below)
    prpacmap_metric='angular' # default 'euclidean', 'angular'
    pr_apply_pca = True # Before PaCMAP
    prpacmap_LR = 1e-5 # Default 1e-3
    prpacmap_batchsize = 512 # Default 1024
    prpacmap_NumEpochs = 500 # Default 450
    prpacmap_weight_schedule = custom_paramrep_weight_schedule
    prpacmap_NN = 10 # Default 10
    prpacmap_n_MN = 20 # Default 20
    prpacmap_n_FP = 5 # Default 5
    prpacmap_num_workers = 8

    # PaCMAP Settings (unless random override below)
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
    apply_pca = True # Before PaCMAP
    pacmap_LR = 0.1 # 0.1 #0.05
    pacmap_NumIters = (1500,1500,1500) # (1500,1500,1500)
    pacmap_NN = 10
    pacmap_MN_ratio = 30 # 7 #0.5
    pacmap_FP_ratio = 60 # 11 #2.0

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
    pca_comp=100

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

    # Create paths and directories for saving dim reduction models and outputs
    latent_dir = f"{model_dir}/{latent_subdir}/{win_sec}SecondWindow_{stride_sec}SecondStride" 
    if run_prpacmap: 
        pr_dir = f"{model_dir}/prpacmap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
        if not os.path.exists(pr_dir): os.makedirs(pr_dir)
    if run_pacmap: 
        pacmap_dir = f"{model_dir}/pacmap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
        if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
    if run_phate: 
        phate_dir = f"{model_dir}/phate/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
        if not os.path.exists(phate_dir): os.makedirs(phate_dir)
    if run_umap: 
        umap_dir = f"{model_dir}/umap/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
        if not os.path.exists(umap_dir): os.makedirs(umap_dir)
    if run_kohenen: 
        kohenen_dir = f"{model_dir}/kohenen/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride"
        if not os.path.exists(kohenen_dir): os.makedirs(kohenen_dir)
   
    ### GENERATION DATA ###
    build_filepaths = [] # Collect pacmap build files - i.e. what data is being used to construct data manifold approximator
    for i in range(len(build_strs)):
        dir_curr = f"{latent_dir}/{build_strs[i]}"
        if single_pats == []: build_filepaths = build_filepaths + glob.glob(dir_curr + f'/*.pkl')
        else: # Get all of the patients listing
            num_single_pats = len(single_pats)
            for j in range(num_single_pats):
                pat_curr = single_pats[j]
                build_filepaths = build_filepaths + glob.glob(dir_curr + f'/{single_pats[j]}*.pkl')
    
    assert (build_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (build_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
    build_start_datetimes, build_stop_datetimes = manifold_utilities.filename_to_datetimes([s.split("/")[-1] for s in build_filepaths]) # Get start/stop datetimes
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
        if single_pats == []: eval_filepaths = eval_filepaths + glob.glob(dir_curr + f'/*.pkl')
        else: 
            num_single_pats = len(single_pats)
            for j in range(num_single_pats):
                pat_curr = single_pats[j]
                eval_filepaths = eval_filepaths + glob.glob(dir_curr + f'/{single_pats[j]}*.pkl')

    if eval_filepaths != []: # May not have any eval data available depending on selections
        assert (eval_filepaths[0].split("/")[-1].split("_")[-1] == f"{stride_sec}secStride.pkl") & (eval_filepaths[0].split("/")[-1].split("_")[-2] == f"{win_sec}secWindow") # Double check window and stride are correct based on file naming [HARDCODED]
        eval_start_datetimes, eval_stop_datetimes = manifold_utilities.filename_to_datetimes([s.split("/")[-1] for s in eval_filepaths]) # Get start/stop datetimes
        eval_pat_ids_list = [s.split("/")[-1].split("_")[0] for s in eval_filepaths] # Get the eval pat_ids
        print("Loading all EVAL latent data from files")
        latent_data_windowed_eval = [''] * len(eval_filepaths)  # Load all of the data into system RAM - list of [window, latent_dim]
        for i in range(len(eval_filepaths)):
            with open(eval_filepaths[i], "rb") as f: 
                latent_data_windowed_eval[i] = pickle.load(f)
    else: print(f"WARNING: no evaluation data for {'_'.join(single_pats)} in these categories: {eval_strs}")

    # ##### RANDOM OVERRIDE #####
    # print("RANDOM OVERRIDE!!!")
    # for _ in range(1000):
    #     prpacmap_LR = 10 ** (-random.uniform(1, 7))
    #     prpacmap_batchsize = 2 ** random.randint(4,8)
    #     prpacmap_NN = random.randint(2, 40)
    #     prpacmap_n_MN = random.randint(2, 120)
    #     prpacmap_n_FP = random.randint(2, 120)

    if run_prpacmap:
    ### ParamRepulsor (PR) PaCMAP GENERATION ###
        # Call the subfunction to create/use pr-pacmap and plot
        if single_pats == []: pr_savedir = f"{pr_dir}/all_pats/generation"
        else: pr_savedir = f"{pr_dir}/{'_'.join(single_pats)}/generation"
        axis_20, reducer, hdb, xy_lims = manifold_utilities.prpacmap_subfunction(
            atd_file = atd_file,
            pat_ids_list=build_pat_ids_list,
            latent_data_windowed=latent_data_windowed_generation, 
            start_datetimes_epoch=build_start_datetimes,  
            stop_datetimes_epoch=build_stop_datetimes,
            epoch=epoch, 
            win_sec=win_sec, 
            stride_sec=stride_sec, 
            savedir=pr_savedir,
            FS = FS,

            # Specific to PR-PACMAP
            prpacmap_num_workers=prpacmap_num_workers,
            prpacmap_metric=prpacmap_metric,
            prpacmap_batchsize=prpacmap_batchsize,
            apply_pca=pr_apply_pca,
            prpacmap_LR = prpacmap_LR,
            prpacmap_NumEpochs = prpacmap_NumEpochs,
            prpacmap_weight_schedule=prpacmap_weight_schedule,
            prpacmap_NN = prpacmap_NN,
            prpacmap_n_MN = prpacmap_n_MN,
            prpacmap_n_FP = prpacmap_n_FP,
            HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
            HDBSCAN_min_samples = HDBSCAN_min_samples,
            plot_preictal_color_sec = plot_preictal_color_sec,
            plot_postictal_color_sec = plot_postictal_color_sec,
            **kwargs)

        # # SAVE OBJECTS
        # manifold_utilities.save_prpacmap_objects(
        #     pacmap_dir=pacmap_savedir,
        #     epoch=epoch,
        #     axis=axis_20,
        #     reducer=reducer, 
        #     reducer_MedDim=reducer_MedDim, 
        #     hdb=hdb, 
        #     xy_lims=xy_lims)

        # ## PR-PACMAP EVAL ONLY ### TODO: modify for PR
        # if eval_filepaths != []:
        #     # Call the subfunction to create/use pacmap and plot
        #     if single_pats == []: pacmap_savedir = f"{pacmap_dir}/all_pats/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_eval"
        #     else: pacmap_savedir = f"{pacmap_dir}/{'_'.join(single_pats)}/nn{pacmap_NN}_mn{pacmap_MN_ratio}_fp{pacmap_FP_ratio}_lr{pacmap_LR}/pacmap_eval"
        #     manifold_utilities.pacmap_subfunction(
        #         atd_file=atd_file,
        #         pat_ids_list=eval_pat_ids_list,
        #         latent_data_windowed=latent_data_windowed_eval, 
        #         start_datetimes_epoch=eval_start_datetimes,  
        #         stop_datetimes_epoch=eval_stop_datetimes,
        #         epoch=epoch, 
        #         win_sec=win_sec, 
        #         stride_sec=stride_sec, 
        #         savedir=pacmap_savedir,
        #         FS = FS,
        #         apply_pca=apply_pca,
        #         pca_comp=pca_comp,
        #         exclude_self_pat=exclude_self_pat,
        #         pacmap_MedDim_numdims = pacmap_MedDim_numdims,
        #         pacmap_LR = pacmap_LR,
        #         pacmap_NumIters = pacmap_NumIters,
        #         pacmap_NN = pacmap_NN,
        #         pacmap_MN_ratio = pacmap_MN_ratio,
        #         pacmap_FP_ratio = pacmap_FP_ratio,
        #         HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
        #         HDBSCAN_min_samples = HDBSCAN_min_samples,
        #         plot_preictal_color_sec = plot_preictal_color_sec,
        #         plot_postictal_color_sec = plot_postictal_color_sec,
        #         xy_lims = xy_lims,
        #         premade_prPaCMAP = reducer,
        #         premade_prPaCMAP_MedDim = reducer_MedDim,
        #         premade_HDBSCAN = hdb,
        #         **kwargs)


    # for _ in range(1000):
    #     pacmap_LR = random.uniform(1e-3, 0.5)
    #     papacmap_NN = random.randint(3,50)
    #     pacmap_MN_ratio = random.uniform(0.1, 20.0)
    #     pacmap_FP_ratio = random.uniform(0.1, 20.0)

    if run_pacmap:
        ### PACMAP GENERATION ###
        # Call the subfunction to create/use pacmap and plot
        if single_pats == []: pacmap_savedir = f"{pacmap_dir}/all_pats/generation"
        else: pacmap_savedir = f"{pacmap_dir}/{'_'.join(single_pats)}/generation"
        axis_20, reducer, hdb, xy_lims = manifold_utilities.pacmap_subfunction(
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

        # SAVE OBJECTS (# TODO: Save currently broken after new conda env, saying "pacmap does not have function save")
        manifold_utilities.save_pacmap_objects(
            pacmap_dir=pacmap_savedir,
            epoch=epoch,
            axis=axis_20,
            reducer=reducer, 
            hdb=hdb, 
            xy_lims=xy_lims)

        ## PACMAP EVAL ONLY ###
        if eval_filepaths != []:
            # Call the subfunction to create/use pacmap and plot
            if single_pats == []: pacmap_savedir = f"{pacmap_dir}/all_pats/eval"
            else: pacmap_savedir = f"{pacmap_dir}/{'_'.join(single_pats)}/eval"
            manifold_utilities.pacmap_subfunction(
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
                pacmap_LR = None,
                pacmap_NumIters = None,
                pacmap_NN = None,
                pacmap_MN_ratio = None,
                pacmap_FP_ratio = None,
                HDBSCAN_min_cluster_size = None,
                HDBSCAN_min_samples = None,
                plot_preictal_color_sec = plot_preictal_color_sec,
                plot_postictal_color_sec = plot_postictal_color_sec,
                xy_lims = xy_lims,
                premade_PaCMAP = reducer,
                premade_HDBSCAN = hdb,
                **kwargs)

    if run_umap:
        ### UMAP GENERATION ###

        # Call the subfunction to create/use umap and plot
        if single_pats == []: umap_savedir = f"{umap_dir}/all_pats/umap_generation"
        else: umap_savedir = f"{umap_dir}/{'_'.join(single_pats)}/umap_generation"
        axis_00, reducer, hdb, xy_lims = manifold_utilities.umap_subfunction(
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

    if run_phate:
        ### PHATE GENERATION ###
        if single_pats == []: phate_savedir = f"{phate_dir}/all_pats/phate_gen"
        else: phate_savedir = f"{phate_dir}/{'_'.join(single_pats)}/phate_gen"

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
        phate_ax20, phate, phate_hdb, phate_xy_lims = manifold_utilities.phate_subfunction(
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
        # manifold_utilities.save_phate_objects(
        #     savedir=phate_savedir,
        #     epoch=epoch,
        #     axis=phate_ax20,
        #     phate=phate, 
        #     hdb=hdb, 
        #     xy_lims=xy_lims)

        ### PHATE EVAL ###
        # TODO
        # if eval_filepaths != []:

    if run_kohenen:
        ### Run the Self Organizing Maps (SOM) Algorith
        if single_pats == []: kohenen_savedir = f"{kohenen_dir}/all_pats/generation"
        else: kohenen_savedir = f"{kohenen_dir}/{'_'.join(single_pats)}/generation"
        axes, som = manifold_utilities.kohenen_subfunction_pytorch(
            atd_file = atd_file,
            pat_ids_list=build_pat_ids_list,
            latent_data_windowed=latent_data_windowed_generation, 
            start_datetimes_epoch=build_start_datetimes,  
            stop_datetimes_epoch=build_stop_datetimes,
            epoch=epoch, 
            win_sec=win_sec, 
            stride_sec=stride_sec, 
            savedir=kohenen_savedir,
            som_batch_size=som_batch_size,
            som_lr=som_lr,
            som_epochs=som_epochs,
            som_gridsize=som_gridsize,
            som_lr_epoch_decay=som_lr_epoch_decay,
            som_sigma=som_sigma,
            som_sigma_epoch_decay=som_sigma_epoch_decay,
            FS = FS,
            HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size,
            HDBSCAN_min_samples = HDBSCAN_min_samples,
            plot_preictal_color_sec = plot_preictal_color_sec,
            plot_postictal_color_sec = plot_postictal_color_sec,
            **kwargs)


    if run_histo:
        ### HISTOGRAM LATENT ###
        # Generation data
        print("Histogram on generation data")
        if single_pats == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
        else: histo_dir = f"{model_dir}/histo_latent/{'_'.join(single_pats)}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_generation"
        if not os.path.exists(histo_dir): os.makedirs(histo_dir)
        manifold_utilities.histogram_latent(
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
            if single_pats == []: histo_dir = f"{model_dir}/histo_latent/all_pats/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
            else: histo_dir = f"{model_dir}/histo_latent/{'_'.join(single_pats)}/Epoch{epoch}/{win_sec}SecondWindow_{stride_sec}SecondStride/histo_eval"
            if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir)
            manifold_utilities.histogram_latent(
                pat_ids_list=eval_pat_ids_list,
                latent_data_windowed=latent_data_windowed_eval, 
                start_datetimes_epoch=eval_start_datetimes,  
                stop_datetimes_epoch=eval_stop_datetimes,
                epoch=epoch, 
                win_sec=win_sec, 
                stride_sec=stride_sec, 
                savedir=histo_dir,
                FS = FS)



