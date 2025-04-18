import pickle, numpy as np, glob, os, random
import matplotlib.pyplot as plt

from utilities import utils_functions


if __name__ == "__main__":

    num_rand_dims = 10
    file_windowsecs = 1
    file_stridesecs = 1
    source_dir = f'/media/glommy1/tornados/bse_inference/sheldrake_epoch1138/latent_files/{file_windowsecs}SecondWindow_{file_stridesecs}SecondStride'
    filename_base = None 
    rewin_windowsecs = 64
    rewin_strideseconds = 4

    files = glob.glob(f'{source_dir}/*.pkl')

    selected_file = None

    if filename_base is not None:
        target_filename = filename_base + '.pkl'
        for file_path in files:
            if os.path.basename(file_path) == target_filename:
                selected_file = file_path
                break
        if selected_file is None:
            print(f"Warning: Filename '{target_filename}' not found in '{source_dir}'. Selecting a random file.")
            selected_file = random.choice(files) if files else None
    else:
        selected_file = random.choice(files) if files else None
        pat_id = selected_file.split("/")[-1].split("_")[0]

    if selected_file:
        print(f"Loading file: {selected_file}")
        try:
            with open(selected_file, 'rb') as f:
                loaded_data = pickle.load(f)
            print("File loaded successfully.")
            if isinstance(loaded_data, dict):
                print("Loaded data is a dictionary.")
            else:
                print("Warning: Loaded data is not a dictionary.")
                print(f"Loaded data type: {type(loaded_data)}")
        except FileNotFoundError:
            print(f"Error: File not found at '{selected_file}'.")
        except pickle.UnpicklingError as e:
            print(f"Error: Failed to unpickle file '{selected_file}'. {e}")
    else:
        print(f"No .pkl files found in '{source_dir}', so no file was loaded.")


# ------------------- Data Rewindowing and Plotting -------------------
if loaded_data is not None and isinstance(loaded_data, dict):
    try:
        ww_means = loaded_data['windowed_weighted_means']
        ww_logvars = loaded_data['windowed_weighted_logvars']
        w_mogpreds = loaded_data['windowed_mogpreds']

        print(f"Original shape of ww_means: {ww_means.shape}")
        print(f"Original shape of ww_logvars: {ww_logvars.shape}")
        print(f"Original shape of w_mogpreds: {w_mogpreds.shape}")

        # Rewindow the data
        rewin_means, rewin_logvars, rewin_mogpreds = utils_functions.rewindow_data(
            ww_means, ww_logvars, w_mogpreds, file_windowsecs, file_stridesecs, rewin_windowsecs, rewin_strideseconds)

        print(f"Rewindowed shape of rewin_means: {rewin_means.shape}")
        print(f"Rewindowed shape of rewin_logvars: {rewin_logvars.shape}")
        print(f"Rewindowed shape of rewin_mogpreds: {rewin_mogpreds.shape}")
        num_windows = rewin_means.shape[0]
        latent_dim = rewin_means.shape[1]
        num_mog_components = rewin_mogpreds.shape[1]
        # Create the figure and subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        plt.style.use('seaborn-v0_8-darkgrid')
        colors = plt.cm.viridis(np.linspace(0, 1, 10))  # Get 10 distinct colors

        # Row 1: rewin_means
        random_latent_dims = np.random.choice(latent_dim, size=min(10, latent_dim), replace=False)
        for i, ld_idx in enumerate(random_latent_dims):
            axs[0].plot(range(num_windows), rewin_means[:, ld_idx], label=f'Latent Dim {ld_idx}', color=colors[i % 10])
        axs[0].set_title('Rewindowed Weighted Means')
        axs[0].set_xlabel('Window')
        axs[0].set_ylabel('Value')
        axs[0].legend(fontsize='small')

        # Row 2: rewin_logvars
        for i, ld_idx in enumerate(random_latent_dims):
            axs[1].plot(range(num_windows), rewin_logvars[:, ld_idx], label=f'Latent Dim {ld_idx}', color=colors[i % 10])
        axs[1].set_title('Rewindowed Weighted Logvars')
        axs[1].set_xlabel('Window')
        axs[1].set_ylabel('Value')
        axs[1].legend(fontsize='small')

        # Row 3: rewin_mogpreds (all components)
        for i in range(num_mog_components):
            axs[2].plot(range(num_windows), rewin_mogpreds[:, i], label=f'Component {i}')
        axs[2].set_title('Rewindowed MoG Predictions (All Components)')
        axs[2].set_xlabel('Window')
        axs[2].set_ylabel('Probability')
        axs[2].legend(fontsize='small')

        fig.suptitle(pat_id)
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(f"Error during plotting: Key '{e}' not found in loaded data.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
else:
    print("No data loaded or loaded data is not a dictionary. Skipping plotting.")