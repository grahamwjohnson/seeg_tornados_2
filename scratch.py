from utilities import utils_functions

model_dir = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/10pats/trained_models/dataset_train80.0_val20.0/spoonbill_Sat_Jan_18_18_16_22_2025'
latent_subdir = '/latent_files/Epoch119'


pacmap_build_strs = ['train', "valfinetune"]
pacmap_eval_strs = ['train', "valfinetune", "valunseen"]

win_sec = 10
stride_sec = 5

kwargs={'dummy_var': 42}

utils_functions.pacmap(
    epoch=9999,
    win_sec=win_sec,
    stride_sec=stride_sec,
    model_dir=model_dir,
    latent_subdir=latent_subdir,
    pacmap_build_strs=pacmap_build_strs, 
    pacmap_eval_strs=pacmap_eval_strs, 
    delete_latent_files=False, 
    **kwargs)
