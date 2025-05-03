#!/home/ghassanmakhoul/miniconda3/envs/tornado/bin/python
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import math
import ToroidalSOM
import pdb
import pickle

def get_bse(bse_codename, **kwargs):
    torch.hub.set_dir('./.torch_hub_cache') # Set a local cache directory for testing

    # Load the BSE model with pretrained weights from GitHub
    checkpoint = torch.hub.load(
        'grahamwjohnson/seeg_tornados_2',
        'load_lbm',
        codename=bse_codename,
        pretrained=True,
        load_bse=True, 
        load_bsp=False,
        load_som=True,
        trust_repo='check',
        force_reload=True
    )
    return checkpoint


def load_kohonen(som_precomputed_path, som_device):
    print(f"Loading Toroidal SOM pretrained weights from FILE: {som_precomputed_path}")
    checkpoint = torch.load(som_precomputed_path, map_location=torch.device(som_device))

    # Retrieve hyperparameters
    grid_size = som_gridsize = checkpoint['grid_size']
    input_dim = checkpoint['input_dim']
    lr = checkpoint['lr']
    sigma = checkpoint['sigma']
    lr_epoch_decay = checkpoint['lr_epoch_decay']
    sigma_epoch_decay = checkpoint['sigma_epoch_decay']
    sigma_min = checkpoint['sigma_min']
    epoch = checkpoint['epoch']
    batch_size = checkpoint['batch_size']

    # Create Toroidal SOM instance with same parameters
    som = ToroidalSOM(grid_size=(grid_size, grid_size), input_dim=input_dim, batch_size=batch_size,
                        lr=lr, lr_epoch_decay=lr_epoch_decay, sigma=sigma,
                        sigma_epoch_decay=sigma_epoch_decay, sigma_min=sigma_min, device=som_device)

    # Load weights
    som.load_state_dict(checkpoint['model_state_dict'])
    som.weights = checkpoint['weights']
    som.reset_device(som_device)

    print(f"Toroidal SOM model loaded from {som_precomputed_path}")

    return som, checkpoint

def load_kohonen(som_precomputed_path, som_device):
    print(f"Loading Toroidal SOM pretrained weights from FILE: {som_precomputed_path}")
    checkpoint = torch.load(som_precomputed_path, map_location=torch.device(som_device))

    # Retrieve hyperparameters
    grid_size = som_gridsize = checkpoint['grid_size']
    input_dim = checkpoint['input_dim']
    lr = checkpoint['lr']
    sigma = checkpoint['sigma']
    lr_epoch_decay = checkpoint['lr_epoch_decay']
    sigma_epoch_decay = checkpoint['sigma_epoch_decay']
    sigma_min = checkpoint['sigma_min']
    epoch = checkpoint['epoch']
    batch_size = checkpoint['batch_size']

    # Create Toroidal SOM instance with same parameters
    som = ToroidalSOM(grid_size=(grid_size, grid_size), input_dim=input_dim, batch_size=batch_size,
                        lr=lr, lr_epoch_decay=lr_epoch_decay, sigma=sigma,
                        sigma_epoch_decay=sigma_epoch_decay, sigma_min=sigma_min, device=som_device)

    # Load weights
    som.load_state_dict(checkpoint['model_state_dict'])
    som.weights = checkpoint['weights']
    som.reset_device(som_device)

    print(f"Toroidal SOM model loaded from {som_precomputed_path}")

    return som, checkpoint
def get_som_rowcol(data, som):
    """Helper to run a batch of data through the SOM and get (row, col) coordinates.

    Args:
        data (np.ndarray or torch.Tensor): The input data.
        som (torch_som.SelfOrganizingMap): The Self-Organizing Map object.

    Returns:
        np.ndarray: An array of shape (data.shape[0], 2) containing the (row, col)
                    coordinates of the best matching unit for each data point.
    """
    som_rowcol = np.zeros((data.shape[0], 2), dtype=np.int32)
    num_samples = data.shape[0]
    batch_size = som.batch_size
    device = som.device

    for i in range(0, num_samples, batch_size):
        batch = data[i:i + batch_size]
        if isinstance(batch, np.ndarray):
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        elif isinstance(batch, torch.Tensor):
            batch_tensor = batch.float().to(device)
        else:
            raise TypeError(f"Input 'data' must be a numpy array or a torch tensor, but got {type(data)}.")

        bmu_rows, bmu_cols = som.find_bmu(batch_tensor)
        bmu_rows_np, bmu_cols_np = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()
        som_rowcol[i:i + batch_size, 0] = bmu_rows_np
        som_rowcol[i:i + batch_size, 1] = bmu_cols_np
    return som_rowcol

# pdb.set_trace()
codename = "sheldrake"
print(f"Loading Toroidal SOM pretrained weights from FILE: {codename}")
bse, som,bsp = get_bse(codename)



som.reset_device(0)
som.batch_size = 10
print("RESET Device\n\n")
print(f"Toroidal SOM model loaded from {codename}: \n{som}")


with open("../data/Train45_allDataGathered_subsampleFileFactor1_64secWindow_32secStride.pkl", 'rb') as f:
    bse_data_dict = pickle.load(f)

# NOTE: ww in the keys means that it has already been windowed and weighted by the MOG preds already
# NOTE: windowing is based on the file name
# NOTE: for ww_means_allfiles (N_lilpickles, n_time-points averaged via windowing,1024 is latent dim space )
# NOTE: each lil pickle's start datetime as a datetime object
# NOTE: Can use all_times_.csv to ascertain pre-ictal-ness
# NOTE: Can use time stamps from sleep_staging for sleep periods

# SOM Inference on all data in batches
# all_means = bse_data_dict['ww_means_allfiles'] # reshaped to shape_dim 0 x shape_dim 1
latent_means_input = torch.rand((100,1024))

som_row_col = get_som_rowcol(latent_means_input, som)
print(som_row_col)


#NOTE: need to get the best matching unit out, that's a row and a column

#NOTE: can plot with squares but lines will be jagged
# Pre-ictal visualization and ictal

# log transformed the hit map, 
# NOTE: Validation data clumps more than the training data, could mean that 
# validation is less similar to the gamut of BMUs on the toroid.
# # can check hitmap for validation to see that most "hit" areas are not just pre-ictal
# NOTE: consider coloring seizures at the end of all this

# NOTE: BSE trained with SPES, but SOM not trained with any SPES data

#NOTE: Goals
# 1. Sleep
# 2. Pre-Ictal
#       a. 4 hour pre-ictal
#       b. down to 1 hour pre-ictal
# 3. Seizures
# 4
