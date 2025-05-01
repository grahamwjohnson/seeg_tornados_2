import torch
from models.BSE import BSE
# from models.BSP import BSP
from models.ToroidalSOM import ToroidalSOM

dependencies = ['torch', 'numpy']

CONFIGS = {
    'sheldrake': {
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

        # Kohonen/SOM Params
        'som_pca_init': False,
        'reduction': 'mean',
        'som_device': 'cpu',
        'som_epochs': 30,
        'som_batch_size': 64,
        'som_lr': 0.5,
        'som_lr_min': 0.01,
        'som_lr_epoch_decay': (0.01 / 0.5)**(1 / 30),
        'som_gridsize': 128,
        'som_sigma': 0.3 * 128,
        'som_sigma_min': 1.0,
        'som_sigma_epoch_decay': (1 / 0.3 * 128)**(1 / 30),

        # BSP Params


        # Weight files
        'bse_weight_file': 'bse_weights.pth',
        'som_dict_file': 'som_dict.pth',
        'bsp_weight_file': 'bsp_weights.pth',
        'release_tag': 'v0.8-alpha'
    }
}

def _load_models(codename='sheldrake', pretrained=True, load_bse=True, load_som=True, load_bsp=True, **kwargs):
    """
    Loads the BSE model & BSP model with specified configuration and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained BSE & BSP models with the specified configuration.
    """
    if codename not in CONFIGS:
        raise ValueError(f"Codenam '{codename}' not found in available configurations: {list(CONFIGS.keys())}")

    config = CONFIGS[codename].copy()
    config.update(kwargs)  # Override with any user-provided kwargs


    # *** Brain-State Embedder (BSE) ***

    if load_bse:
        bse = BSE(**config)

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
            print(f"No BSE weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized BSE model.")

    # *** Kohonen/Self-Organizing Map (SOM) ***
    if load_som:
        som = ToroidalSOM(grid_size=(config['som_gridsize'], config['som_gridsize']), input_dim=config['latent_dim'], batch_size=config['som_batch_size'],
                    lr=config['som_lr'], lr_epoch_decay=config['som_lr_epoch_decay'], sigma=config['som_sigma'],
                    sigma_epoch_decay=config['som_sigma_epoch_decay'], sigma_min=config['som_sigma_min'], device=config['som_device'])
        
        if pretrained and config.get('som_dict_file') and config.get('release_tag'):
            weight_file = config['som_dict_file']
            release_tag = config['release_tag']
            checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
            try:
                checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url)   
                som.load_state_dict(checkpoint['model_state_dict'])
                som.weights = checkpoint['weights']
                som.reset_device(config['som_device'])
            except Exception as e:
                print(f"Error loading pretrained SOM weights for codename '{codename}': {e}")
                print("Continuing with randomly initialized SOM model.")
        elif pretrained:
            print(f"No SOM weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized SOM model.")

    else:
        som = None

    # *** Brain-Sate Predictor (BSP) ***
    bsp = None
    # if load_bsp:
    #     bsp = BSP(**config)

    #     # BSP: Load pretrained weights if requested
    #     if pretrained and config.get('bsp_weight_file') and config.get('release_tag'):
    #         weight_file = config['bsp_weight_file']
    #         release_tag = config['release_tag']
    #         checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/download/{release_tag}/{weight_file}'
    #         try:
    #             state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
    #             bsp.load_state_dict(state_dict)
    #         except Exception as e:
    #             print(f"Error loading pretrained weights for codename '{codename}': {e}")
    #             print("Continuing with randomly initialized model.")
    #     elif pretrained:
    #         print(f"No weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized model.")

    return bse, som, bsp


def load_lbm(codename='sheldrake', pretrained=True, load_bse=True, load_som=True, load_bsp=True, **kwargs):
    """
    Loads the BSE, Kohonen, & BSP models with a specific training run's configuration
    and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained BSE, Kohonen, & BSP models with the specified configuration.
    """
    return _load_models(codename=codename, pretrained=pretrained, load_bse=load_bse, load_som=load_som, load_bsp=load_bsp, **kwargs)


