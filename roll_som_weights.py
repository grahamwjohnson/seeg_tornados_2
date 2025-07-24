import torch
import os
from models.ToroidalSOM import ToroidalSOM  

def load_kohonen(som_precomputed_path, som_device):
    print(f"Loading Toroidal SOM pretrained weights from FILE: {som_precomputed_path}")
    checkpoint = torch.load(som_precomputed_path)

    som = ToroidalSOM(
        grid_size=(checkpoint['grid_size'], checkpoint['grid_size']),
        input_dim=checkpoint['input_dim'],
        batch_size=checkpoint['batch_size'],
        lr=checkpoint['lr'],
        lr_epoch_decay=checkpoint['lr_epoch_decay'],
        sigma=checkpoint['sigma'],
        sigma_epoch_decay=checkpoint['sigma_epoch_decay'],
        sigma_min=checkpoint['sigma_min'],
        device=som_device
    )

    som.load_state_dict(checkpoint['model_state_dict'])
    som.weights = checkpoint['weights']
    som.reset_device(som_device)

    print(f"Toroidal SOM model loaded from {som_precomputed_path}")
    return som, checkpoint

def roll_and_save_som(som_path, save_dir, device, roll_vertical=0, roll_horizontal=0):
    som, checkpoint = load_kohonen(som_path, device)

    # Perform the rolling
    print(f"Rolling SOM weights: vertical={roll_vertical}, horizontal={roll_horizontal}")
    rolled_weights = torch.roll(som.weights, shifts=(roll_vertical, roll_horizontal), dims=(0, 1))

    # Update the weights
    som.weights = rolled_weights

    # Prepare filename
    base_filename = os.path.splitext(os.path.basename(som_path))[0]
    new_filename = f"{base_filename}_rolled_v{roll_vertical}_h{roll_horizontal}.pt"
    save_path = os.path.join(save_dir, new_filename)

    # Save new checkpoint
    print(f"Saving rolled SOM to {save_path}")
    torch.save({
        'model_state_dict': som.state_dict(),
        'weights': som.weights,
        'grid_size': checkpoint['grid_size'],
        'input_dim': checkpoint['input_dim'],
        'pca': checkpoint['pca'],
        'lr': checkpoint['lr'],
        'cim_kernel_sigma': checkpoint['cim_kernel_sigma'],
        'sigma': checkpoint['sigma'],
        'lr_epoch_decay': checkpoint['lr_epoch_decay'],
        'sigma_epoch_decay': checkpoint['sigma_epoch_decay'],
        'sigma_min': checkpoint['sigma_min'],
        'epoch': checkpoint['epoch'],
        'batch_size': checkpoint['batch_size'],
    }, save_path)

    print(f"Rolled SOM saved successfully to {save_path}")

def main():
    som_precomputed_path = '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_all58/kohonen/64SecondWindow_16SecondStride_Reductionmean/all_pats_64som_LR0.001/GPU0_ToroidalSOM_ObjectDict_smoothsec64_Stride16_subsampleFileFactor1_preictalSec14400_gridsize64_lr0.5with0.9397decay0.001000min_sigma32.0with0.9659decay1min_numfeatures2743020_dims8_batchsize1024_epochs100.pt'  
    save_directory = '/media/glommy1/tornados/bsv_inference/commongonolek_epoch296_sheldrake_epoch1138_all58/kohonen/64SecondWindow_16SecondStride_Reductionmean/all_pats_64som_LR0.001'
    device = "cpu"  # or "cuda"

    roll_vertical = -40   # positive = roll down, negative = roll up
    roll_horizontal = -3  # positive = roll right, negative = roll left

    # Make sure save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Call the function
    roll_and_save_som(
        som_path=som_precomputed_path,
        save_dir=save_directory,
        device=device,
        roll_vertical=roll_vertical,
        roll_horizontal=roll_horizontal
    )

if __name__ == "__main__":
    main()
