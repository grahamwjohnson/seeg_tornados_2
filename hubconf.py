import torch
from models.BSE import BSE, Discriminator
from models.BSP import BSP, BSV
from models.ToroidalSOM_2 import ToroidalSOM_2

dependencies = ['torch', 'numpy']

CONFIGS = {
    'commongonolek_sheldrake': {

        # BSE Params
        'encode_token_samples': 1,
        'padded_channels': 256,
        'transformer_seq_length': 512,
        'max_seq_len': 512,
        'max_batch_size': 8,
        'n_layers': 16,
        'n_heads': 32,
        'multiple_of': 256,
        'ffn_dim_multiplier': 1.0,
        'attention_dropout': 0.1,
        'transformer_start_pos': 0,
        'transformer_dim': 1024,
        'encoder_transformer_activation': "silu",
        'top_dims': 1024,
        'hidden_dims': 1024,
        'latent_dim': 1024,
        'decoder_base_dims': 4096,
        'prior_mog_components': 8,
        'mean_lims': [-5, 5],
        'logvar_lims': [-5, 1],
        'gumbel_softmax_temperature_max': 0.05,
        'diag_mask_buffer_tokens': 16,
        'prior_initial_mean_spread': 3,
        'prior_initial_logvar': -2,
        'gp_sigma': 2.5, 
        'gp_length_scale': 128,
        'crattn_num_heads': 8,
        'crattn_num_layers': 16,
        'crattn_max_seq_len': 1,
        'crattn_dropout': 0.1, 
        'posterior_mogpredictor_hidden_dim_list': [2048, 1024, 512], 
        'posterior_mogpredictor_dropout': 0.1,
        'classifier_hidden_dims': [2048, 1024, 512], 
        'classifier_num_pats': 45, 
        'classifier_dropout': 0.1,

        # Discriminator Params
        'disc_hidden_dims': [4096, 2048, 1024, 512, 256],

        # BSP Params
        'bse2p_chunk_size': 16*2*1024,
        'bse2p_transformer_seq_length': 512,
        'bsp_transformer_seq_length': 32,
        'bsp_latent_dim': 1024,
        'bsp_n_heads': 32,
        'bsp_n_layers': 16,
        'bsp_ffn_dim_multiplier': 0.8,
        'bsp_max_batch_size': 1,
        'bsp_max_seq_len': 32,
        'bsp_transformer_activation': "silu",
        'bsp_attention_dropout': 0.0,
        'bsp_transformer_start_pos': 0,
        'bsp2e_chunk_size': 16*2*1024,
        'bsp2e_transformer_seq_length': 512,

        # BSV Params
        'bsv_dims': [1024, 512, 256, 128, 8],

        # Kohonen/SOM Params
        'som_pca_init': False,
        'reduction': 'mean', # Keep at mean because currently using reparam in SOM training
        'som_epochs': 100,
        'som_batch_size': 1024,
        'som_lr': 0.5,
        'som_lr_min': 0.001, 
        'som_lr_epoch_decay': 0.9397455978,
        'som_gridsize': 64,
        'som_sigma': 32,
        'som_sigma_min': 1,
        'som_sigma_epoch_decay': 0.96593632892,

        # Weight files
        'bse_weight_file': 'bse_weights.pth',
        'disc_weight_file': 'disc_weights.pth',
        'bsp_weight_file': 'bsp_weights.pth',
        'bsv_weight_file': 'bsv_weights.pth',
        'som_file': 'som_file.pth',
        'release_tag': 'v0.8-alpha'
    }
}

def _load_models(codename='commongonolek_sheldrake', gpu_id='cpu', pretrained=True, load_bse=True, load_discriminator=True, load_bsp=True, load_bsv=True, load_som=True, **kwargs):
    """
    Loads the BSE, BSP, BSV, & 2D-PaCMAP model with specified configuration and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained BSE, BSP, BSV, & PaCMAP models with the specified configuration.
    """
    if codename not in CONFIGS:
        raise ValueError(f"Codename '{codename}' not found in available configurations: {list(CONFIGS.keys())}")

    config = CONFIGS[codename].copy()
    config.update(kwargs)  # Override with any user-provided kwargs


    # *** Brain-State Embedder (BSE) ***

    if load_bse:
        bse = BSE(gpu_id=gpu_id, **config)

        if pretrained and config.get('bse_weight_file') and config.get('release_tag'):
            weight_file = config['bse_weight_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
            try:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
                bse.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading pretrained BSE weights for codename '{codename}': {e}")
                print("Continuing with randomly initialized BSE model.")
        elif pretrained:
            print(f"No BSE weight file or release tag specified for BSE codename '{codename}'. Continuing with randomly initialized BSE model.")

    # *** KLD Adversarial Discriminator for BSE posterior vs. prior ***
    disc = None
    if load_discriminator:
        disc = Discriminator(gpu_id=gpu_id, **config)
        
        # Discriinator: Load pretrained weights if requested
        if pretrained and config.get('disc_weight_file') and config.get('release_tag'):
            weight_file = config['disc_weight_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
            try:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
                disc.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading pretrained Disc weights for codename '{codename}': {e}")
                print("Continuing with randomly initialized model.")
        elif pretrained:
            print(f"No weight file or release tag specified for Disc codename '{codename}'. Continuing with randomly initialized model.")


    # *** Brain-Sate Predictor (BSP) ***
    bsp = None
    if load_bsp:
        bsp = BSP(gpu_id=gpu_id, **config)

        # BSP: Load pretrained weights if requested
        if pretrained and config.get('bsp_weight_file') and config.get('release_tag'):
            weight_file = config['bsp_weight_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
            try:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
                bsp.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading pretrained BSP weights for codename '{codename}': {e}")
                print("Continuing with randomly initialized model.")
        elif pretrained:
            print(f"No weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized model.")


    # *** Brain-Sate Visualizer (BSV) ***
    bsv = None
    if load_bsv:
        bsv = BSV(gpu_id=gpu_id, **config)

        # BSV: Load pretrained weights if requested
        if pretrained and config.get('bsv_weight_file') and config.get('release_tag'):
            weight_file = config['bsv_weight_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
            try:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
                bsv.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading pretrained BSV for codename '{codename}': {e}")
                print("Continuing with randomly initialized model.")
        elif pretrained:
            print(f"No weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized model.")


    # *** 2D SOM/Kohonen ***
    som = None
    if load_som:
        try:
            print("Attempting to load pretrained som model for 2D visualization of BSV")
            som_precomputed_path = config['som_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'

            checkpoint = torch.load(som_precomputed_path)

            # Retrieve hyperparameters
            grid_size = som_gridsize = checkpoint['grid_size']
            input_dim = checkpoint['input_dim']
            lr = checkpoint['lr']
            sigma = checkpoint['sigma']
            pca = checkpoint['pca']
            lr_epoch_decay = checkpoint['lr_epoch_decay']
            sigma_epoch_decay = checkpoint['sigma_epoch_decay']
            sigma_min = checkpoint['sigma_min']
            epoch = checkpoint['epoch']
            batch_size = checkpoint['batch_size']
            cim_kernel_sigma = checkpoint['cim_kernel_sigma']

            # Create Toroidal SOM instance with same parameters
            som = ToroidalSOM_2(grid_size=(grid_size, grid_size), input_dim=input_dim, batch_size=batch_size,
                            lr=lr, lr_epoch_decay=lr_epoch_decay, cim_kernel_sigma=cim_kernel_sigma, sigma=sigma,
                            sigma_epoch_decay=sigma_epoch_decay, sigma_min=sigma_min, pca=pca, device='cpu', data_for_init=None)

            # Load weights
            som.load_state_dict(checkpoint['model_state_dict'])
            som.weights = checkpoint['weights']

            print(f"Toroidal SOM model loaded from {som_precomputed_path}")

            # TODO 



        except Exception as e:
            print(f"Error loading som for codename '{codename}': {e}")
            print("Returning empty variable")


    return bse, disc, bsp, bsv, som


def load_lbm(codename='commongonolek_sheldrake', pretrained=True, load_bse=True, load_discriminator=True, load_bsp=True, load_bsv=True, load_som=True, **kwargs):
    """
    Loads the BSE, BSP, BSV, & PaCMAP models with a specific training run's configuration
    and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained BSE, BSP, BSV, & PaCMAP models with the specified configuration.
    """
    return _load_models(codename=codename, pretrained=pretrained, load_bse=load_bse, load_discriminator=load_discriminator, load_bsp=load_bsp, load_bsv=load_bsv, load_som=load_som, **kwargs)


