import torch
from models.GMVAE import GMVAE

dependencies = ['torch', 'numpy']

CONFIGS = {
    'sheldrake': {
        'encode_token_samples': 1,
        'padded_channels': 256,
        'transformer_seq_length': 512,
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
        'weight_file': 'gmvae_weights.pth',
        'release_tag': 'v0.3-alpha'
    }
}

def _load_gmvae(codename='sheldrake', pretrained=True, **kwargs):
    """
    Loads the GM-VAE model with specified configuration and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained GM-VAE model with the specified configuration.
    """
    if codename not in CONFIGS:
        raise ValueError(f"Codenam '{codename}' not found in available configurations: {list(CONFIGS.keys())}")

    config = CONFIGS[codename].copy()
    config.update(kwargs)  # Override with any user-provided kwargs

    # Create model with architecture parameters
    model = GMVAE(
        encode_token_samples=config['encode_token_samples'],
        padded_channels=config['padded_channels'],
        transformer_seq_length=config['transformer_seq_length'],
        transformer_start_pos=config['transformer_start_pos'],
        transformer_dim=config['transformer_dim'],
        encoder_transformer_activation=config['encoder_transformer_activation'],
        top_dims=config['top_dims'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        decoder_base_dims=config['decoder_base_dims'],
        prior_mog_components=config['prior_mog_components'],
        mean_lims=config['mean_lims'],
        logvar_lims=config['logvar_lims'],
        gumbel_softmax_temperature_max=config['gumbel_softmax_temperature_max'],
        diag_mask_buffer_tokens=config['diag_mask_buffer_tokens'],
    )

    # Load pretrained weights if requested
    if pretrained and config.get('weight_file') and config.get('release_tag'):
        weight_file = config['weight_file']
        release_tag = config['release_tag']
        checkpoint_url = f'https://github.com/grahamwjohnson/seeg_tornados_2/releases/tag/{release_tag}/{weight_file}'
        try:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading pretrained weights for codename '{codename}': {e}")
            print("Continuing with randomly initialized model.")
    elif pretrained:
        print(f"No weight file or release tag specified for codename '{codename}'. Continuing with randomly initialized model.")

    return model

def load(codename='sheldrake', pretrained=True, **kwargs):
    """
    Loads the GM-VAE model with a specific training run's configuration
    and optionally pretrained weights.

    Args:
        codename (str): The codename of the training run to load (e.g., 'sheldrake').
        pretrained (bool): If True, returns a model pretrained for the given codename.
        **kwargs: Additional parameters to override default configuration.

    Returns:
        Pretrained GM-VAE model with the specified configuration.
    """
    return _load_gmvae(codename=codename, pretrained=pretrained, **kwargs)


