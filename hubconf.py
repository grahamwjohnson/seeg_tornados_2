import torch
from models.BSE import BSE, Discriminator
from models.BSP import BSP, BSV

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

        # BSP Params
        'bse2p_transformer_dim': 64,
        'bse2p_layers': 8,
        'bse2p_num_heads': 8,
        'bsp2e_ffn_dim_multiplier': 1.0,
        'bsp2e_max_batch_size': 32,
        'bsp2e_hidden_dims': [64,128,256, 512],
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

        # BSV Params
        'bsv_dims': [1024, 512, 256, 128, 8],

        # PaCMAP Params


        # Weight files
        'bse_weight_file': 'bse_weights.pth',
        'disc_weight_file': 'disc_weights.pth',
        'bsp_weight_file': 'bsp_weights.pth',
        'bsv_weight_file': 'bsv_weights.pth',
        'pacmap_base': 'pacmap_2d',
        'release_tag': 'v0.8-alpha'
    }
}

def _load_models(codename='midge_sheldrake', pretrained=True, load_bse=True, load_discriminator=True, load_bsp=True, load_bsv=True, load_pacmap=True, **kwargs):
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
            print(f"No BSE weight file or release tag specified for BSE codename '{codename}'. Continuing with randomly initialized BSE model.")

    # *** KLD Discriminator for post-BSE manifold***
    disc = None
    if load_discriminator:
        disc = Discriminator(**config)
        
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
        bsp = BSP(**config)

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
        bsv = BSV(**config)

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


    # *** 2D PaCMAP ***
    pacmap = None
    if load_pacmap:
        try:
            print("Attempting to load pretrained pacmap model for 2D visualization of BSV")



            # TODO 



        except Exception as e:
            print(f"Error loading pacmap for codename '{codename}': {e}")
            print("Returning empty variable")


    return bse, disc, bsp, bsv, pacmap


def load_lbm(codename='midge_sheldrake', pretrained=True, load_bse=True, load_discriminator=True, load_bsp=True, load_bsv=True, load_pacmap=True, **kwargs):
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
    return _load_models(codename=codename, pretrained=pretrained, load_bse=load_bse, load_discriminator=load_discriminator, load_bsp=load_bsp, load_bsv=load_bsv, load_pacmap=load_pacmap, **kwargs)


