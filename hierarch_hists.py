import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from inference_BSE import bse_export_embeddings, get_bse, load_dataset
from utilities import utils_functions
from utilities import manifold_utilities

import numpy as np
import matplotlib.pyplot as plt

gpu_id = 0
world_size = 1

bse_passes = 256
num_dims_to_plot = 3

ax_lims = [-5,5]

savedir = '/media/glommy1/tornados/bse_inference/sheldrake_epoch1138_train45/tmp_plots'

# Tier 0
big_window_indices = np.arange(0,64)

# Tier 1
tier1_window_size = 64
tier1_stride = 32
max_tier1_chunks = 32

# Read in configuration file 
config_f = 'config.yml'
with open(config_f, "r") as f: kwargs = yaml.load(f,Loader=yaml.FullLoader)
kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs

# Load the inference patients
inference_dataset = load_dataset(gpu_id, **kwargs)

# Load the pretrained Brain-State Embedder (BSE) from GitHub and put on GPU, and initialize DDP
bse = get_bse(**kwargs)
bse = bse.to(gpu_id) 
bse.gpu_id = gpu_id
bse.transformer_encoder.freqs_cis = bse.transformer_encoder.freqs_cis.to(gpu_id)

# Run preprocessed data through BSE and save according to export settings in config.yaml
with torch.inference_mode():
    means_full, logvars_full, mogpreds_full, z_full = bse_export_embeddings(
        world_size=world_size,
        gpu_id=gpu_id,
        bse=bse,
        dataset_curr=inference_dataset,
        dataset_string='bse_inference',
        return_random_file_embeddings = True,
        return_random_file_embeddings_win_size = bse_passes,
        **kwargs)
    
# Pull off gpu and make into numpy array
means_full = means_full.cpu().numpy()
mogpreds_full = mogpreds_full.cpu().numpy()

# Dummy indices
batch_idx = 0
big_window_idx = 0  # Select your target big window
ax_lims = kwargs['mean_lims']

manifold_utilities.plot_mog_histograms(
    savedir, 
    means_full, 
    mogpreds_full, 
    batch_idx=batch_idx, 
    ax_lims=ax_lims,
    num_dims=num_dims_to_plot,

    # Tier 0
    big_window_indices=big_window_indices,
    
    # Tier 1
    window_size=tier1_window_size,
    stride=tier1_stride,
    max_tier1_chunks=max_tier1_chunks
    )

