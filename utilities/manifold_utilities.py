import os
import pacmap
import hdbscan
import matplotlib.pylab as pl
pl.switch_backend('agg')
import datetime
import csv
import matplotlib.gridspec as gridspec
from .latent_plotting import plot_latent
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.ToroidalSOM import ToroidalSOM
from models.ToroidalSOM_2 import ToroidalSOM_2, select_cim_sigma
import torch
# import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
# import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import RegularPolygon
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from matplotlib import colors as mcolors
import heapq
import random
from scipy.interpolate import splprep, splev
import phate

'''
@author: grahamwjohnson

Seperate utilities repository from utils_functions.py to enable a slimmer/simpler conda env for manifolds.py

'''

import heapq
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

import numpy as np

def find_toroidal_low_cost_path(u_matrix, start, goal, undo_log):
    """
    Finds the lowest-cost path between two points on a toroidal U-Matrix by exploring small variations 
    of direct and toroidal paths without using Dijkstra's algorithm.

    Args:
        u_matrix (np.ndarray): 2D array representing the U-Matrix values (costs).
        start (tuple): (x, y) start coordinates (integers).
        goal (tuple): (x, y) goal coordinates (integers).
        max_variations (int): Number of variations to explore.

    Returns:
        path (list): List of (x, y) points from start to goal following the lowest U-Matrix cost.
    """
    rows, cols = u_matrix.shape

    start = (start[1], start[0])  # Switch to (row, col) internally
    goal = (goal[1], goal[0])

    def compute_path_cost(path, u_matrix, undo_log):
        """Compute the total cost of a given path."""
        if undo_log:
            return sum(np.exp(u_matrix[r, c]) for r, c in path)
        else:
            return sum(u_matrix[r, c] for r, c in path)

    def create_path(start, goal, rows, cols, wrap_vertical=False, wrap_horizontal=False):
        path = []
        r1, c1 = start
        r2, c2 = goal

        row_vec = np.arange(0, rows)
        col_vec = np.arange(0, cols)

        # --- Determine vertical move direction and steps ---
        if wrap_vertical:
            # Wrap vertical distance
            # This checks both directions to wrap properly based on whether r1 > r2
            if r1 > r2:
                row_steps = (r2 - r1) % rows  # Moving down
                vert_dir = 1  # Move down (positive direction)
            else:
                row_steps = (r1 - r2) % rows  # Moving up
                vert_dir = -1  # Move up (negative direction)
        else:
            # No wrapping for vertical movement
            row_steps = abs(r2 - r1)
            vert_dir = 1 if r2 > r1 else -1

        # --- Determine horizontal move direction and steps ---
        if wrap_horizontal:
            # Wrap horizontal distance
            # This checks both directions to wrap properly based on whether c1 > c2
            if c1 > c2:
                col_steps = (c2 - c1) % cols  # Moving right
                horz_dir = 1  # Move right (positive direction)
            else:
                col_steps = (c1 - c2) % cols  # Moving left
                horz_dir = -1  # Move left (negative direction)
        else:
            # No wrapping for horizontal movement
            col_steps = abs(c2 - c1)
            horz_dir = 1 if c2 > c1 else -1

        # --- Gradual stepping rates ---
        total_steps = max(row_steps, col_steps)
        row_step_every = total_steps / row_steps if row_steps != 0 else float('inf')
        col_step_every = total_steps / col_steps if col_steps != 0 else float('inf')

        # --- Counters ---
        row_counter = 0
        col_counter = 0

        for _ in range(int(total_steps)):
            # Move row if necessary
            if row_counter >= row_step_every:
                r1 = (r1 + vert_dir) % rows
                row_counter -= row_step_every

            # Move column if necessary
            if col_counter >= col_step_every:
                c1 = (c1 + horz_dir) % cols
                col_counter -= col_step_every

            path.append((r1, c1))

            row_counter += 1
            col_counter += 1

        # Add last point to path
        path.append(goal)

        return path
    
    # Step 1: Generate the direct, vertical toroidal, horizontal toroidal, and double-wrapping paths
    direct_path = create_path(start, goal, rows, cols, wrap_vertical=False, wrap_horizontal=False)
    vertical_toroidal_path = create_path(start, goal, rows, cols, wrap_vertical=True, wrap_horizontal=False)
    horizontal_toroidal_path = create_path(start, goal, rows, cols, wrap_vertical=False, wrap_horizontal=True)
    double_wrapping_path = create_path(start, goal, rows, cols, wrap_vertical=True, wrap_horizontal=True)

    # Step 2: Evaluate all variations and select the lowest-cost path
    paths = [direct_path, vertical_toroidal_path, horizontal_toroidal_path, double_wrapping_path]
    costs = [compute_path_cost(path, u_matrix, undo_log) for path in paths]
    min_cost_index = np.argmin(costs)

    # Switch back to row/col 
    best_path = paths[min_cost_index]
    best_path = [(x[1], x[0]) for x in best_path]
    
    best_cost = costs[min_cost_index]

    return best_path, best_cost

def split_on_wraps(points, rows, cols):
    """Split points into segments wherever a toroidal wrap occurs."""
    segments = []
    current = [points[0]]
    for i in range(1, len(points)):
        r1, c1 = points[i - 1]
        r2, c2 = points[i]
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        if dr > rows // 2 or dc > cols // 2:
            segments.append(np.array(current))
            current = [points[i]]
        else:
            current.append(points[i])
    if current:
        segments.append(np.array(current))
    return segments

def smooth_hex_segment(hex_points, smoothing_factor):
    """Apply spline smoothing to a sequence of (x, y) points."""
    if len(hex_points) < 4:
        return hex_points  # not enough points to smooth
    x, y = hex_points[:, 0], hex_points[:, 1]
    try:
        tck, _ = splprep([x, y], s=smoothing_factor)
        u_fine = np.linspace(0, 1, len(x) * 10)
        x_smooth, y_smooth = splev(u_fine, tck)
        return np.vstack([x_smooth, y_smooth]).T
    except:
        return hex_points  # fallback if spline fails

def create_colored_line(ax, points, color_start, color_end, label, u_matrix, undo_log, smoothing_factor, ball_at_start=False):
    """Draw smoothed, wrap-aware hex lines with a color gradient and a final star."""
    if len(points) < 2:
        return

    rows, cols = u_matrix.shape

    # Reconstruct full toroidal path
    full_path = []
    for i in range(len(points) - 1):
        path_segment, _ = find_toroidal_low_cost_path(u_matrix, points[i], points[i + 1], undo_log)
        full_path.extend(path_segment[:-1])
    if full_path == []:
        full_path = [points[-1]]
    elif tuple(full_path[-1]) != tuple(points[-1]):
        full_path.append(points[-1])
    full_path = np.array(full_path)

    # Split by toroidal wraps
    path_segments = split_on_wraps(full_path, rows, cols)

    # Convert to hex coords, smooth each segment
    smooth_segments = []
    for seg in path_segments:
        hex_seg = np.array([hex_grid_position(p[1], p[0]) for p in seg])  # col, row -> x, y
        smooth_seg = smooth_hex_segment(hex_seg, smoothing_factor)
        smooth_segments.append(smooth_seg)

    # Generate color gradient across total path
    total_len = sum(len(seg) for seg in smooth_segments)
    rgb_start = np.array(mcolors.to_rgb(color_start))
    rgb_end = np.array(mcolors.to_rgb(color_end))

    colored_segments = []
    colors = []
    idx = 0
    for seg in smooth_segments:
        for i in range(len(seg) - 1):
            alpha = idx / (total_len - 1) if total_len > 1 else 0
            rgb = rgb_start + (rgb_end - rgb_start) * alpha
            colored_segments.append([seg[i], seg[i + 1]])
            colors.append((*rgb, 1.0))
            idx += 1

    lc = LineCollection(colored_segments, colors=colors, linewidth=2)
    ax.add_collection(lc)

    # Plot final star at end point
    final_point = full_path[-1]
    x, y = hex_grid_position(final_point[1], final_point[0])
    ax.plot(x, y,
            marker='*', color=color_end,
            markersize=12,
            markeredgecolor=None, markeredgewidth=0.4,
            label=label)
    
    # Place a ball at start of segment if desired
    if ball_at_start:
        start_point = full_path[0]
        x, y = hex_grid_position(start_point[1], start_point[0])
        ax.plot(x, y,
                marker='.', color=color_start,
                markersize=12,
                markeredgecolor=None, markeredgewidth=0.4,
                label=label)

def hex_grid_position(row, col, radius=1.0):
    """Returns (x, y) position in the hexagonal grid."""
    width = 1.5 * radius
    height = np.sqrt(3) * radius
    x = col * width
    y = row * height + (col % 2) * (height / 2)
    return x, y

def plot_trajectory_on_umatrix(ax, context, ground_truth, predictions, som_nodes, u_matrix, undo_log, smoothing_factor):
    """
    Plots the trajectory of points on an existing U-matrix plot, following the lowest-cost toroidal paths.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object of the existing U-matrix plot.
        context (list): List of node indices representing the context trajectory.
        ground_truth (list): List of node indices representing the ground truth trajectory.
        predictions (list): List of node indices representing the predicted trajectory.
        som_nodes (np.ndarray): Array of SOM node coordinates (n_nodes x 2).
        u_matrix (np.ndarray): 2D array of U-Matrix values (for calculating lowest-cost paths).
    """

    def get_node_coordinates(node_indices, som_nodes):
        """Retrieves (x, y) coordinates from som_nodes based on node indices."""
        return [som_nodes[i[1] + i[0] * int(np.sqrt(len(som_nodes)))] for i in node_indices]

    # Map trajectories to (x, y) coordinates
    context_points = get_node_coordinates(context, som_nodes)
    ground_truth_points = get_node_coordinates(ground_truth, som_nodes)
    prediction_points = get_node_coordinates(predictions, som_nodes)

    # Define color gradients and plot
    create_colored_line(ax, context_points, '#959ca3', '#49464d', label='Context', u_matrix=u_matrix, undo_log=undo_log, smoothing_factor=smoothing_factor, ball_at_start=True)  # Gray
    create_colored_line(ax, ground_truth_points, '#21473a', '#7cbf98', label='Ground Truth', u_matrix=u_matrix, undo_log=undo_log, smoothing_factor=smoothing_factor) # Blue
    create_colored_line(ax, prediction_points, '#164370', '#4a91d9', label='Prediction', u_matrix=u_matrix, undo_log=undo_log, smoothing_factor=smoothing_factor) # Green

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_xlabel('SOM Node X')
    ax.set_ylabel('SOM Node Y')
    ax.set_title('Predicted Trajectory: U-Matrix with Pre-Ictal Overlay')

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

def plot_kohonen_prediction(gpu_id, save_dir, som, plot_data_path, context, ground_truth_future, predictions, undo_log, smoothing_factor, epoch, batch_idx, pat_id, preictal_overlay_thresh=0.25):
    """Plots Kohonen/SOM predictions on top of a U-Matrix + Pre-Ictal overlay."""
    # Process context, ground truth, and predictions separately
    context_rowcol = get_som_rowcol(context, som)
    gt_rowcol = get_som_rowcol(ground_truth_future, som)
    pred_rowcol = get_som_rowcol(predictions, som)

    # Load underlay plotting data
    with open(plot_data_path, "rb") as f:
        file_data = pickle.load(f)
    grid_size = som.grid_size
    u_matrix_hex = file_data['u_matrix_hex']
    overlay_preictal = file_data['rescale_preictal_smoothed']

    # Create figure
    fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

    # Plot U-Matrix background
    plot_hex_grid(ax_overlay, u_matrix_hex, "Predicted Trajectory: U-Matrix with Pre-Ictal Overlay",
                  cmap_str='bone_r', 
                  vmin=np.min(u_matrix_hex),
                  vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # Overlay Pre-Ictal density
    rows, cols = overlay_preictal.shape
    radius = 1.0
    height = np.sqrt(3) * radius
    cmap_overlay = cm.get_cmap('flare')
    norm_overlay = pl.Normalize(vmin=0.0, vmax=1.0)

    for i in range(rows):
        for j in range(cols):
            x = j * 1.5 * radius
            y = i * height + (j % 2) * (height / 2)
            face_color = cmap_overlay(norm_overlay(overlay_preictal[i, j]))
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                              orientation=np.radians(30),
                                              facecolor=face_color, alpha=0.7,
                                              edgecolor=None, linewidth=0)
            if overlay_preictal[i, j] >= preictal_overlay_thresh:
                ax_overlay.add_patch(hexagon)

    # Colorbar for overlay
    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    pl.colorbar(sm_overlay, ax=ax_overlay, label="Pre-Ictal Density (Clipped & Smoothed)")

    # SOM nodes: (x, y) coordinates
    som_nodes = np.array([(j, i) for i in range(grid_size[0]) for j in range(grid_size[1])])

    # Plot trajectories
    plot_trajectory_on_umatrix(ax_overlay, 
        context_rowcol.tolist(), 
        gt_rowcol.tolist(), 
        pred_rowcol.tolist(), 
        som_nodes,
        u_matrix=u_matrix_hex,
        undo_log=undo_log,
        smoothing_factor=smoothing_factor)

    # Save figure
    savename_overlay = save_dir + f"/kohonen_predictions_epoch{epoch}_batch{batch_idx}_{pat_id}_GPU{gpu_id}.jpg"
    pl.savefig(savename_overlay, dpi=300)
    pl.close('all')

def get_dataset_hours(win_sec, stride_sec, latent_means_windowed):
    """
    Calculates the total hours of the dataset represented by the windowed data,
    accurately accounting for window overlap or gaps (consistent number of windows per file).

    Args:
        win_sec (int): The duration of each window in seconds.
        stride_sec (int): The stride between the start of consecutive windows in seconds.
        latent_means_windowed (list or np.ndarray): A 3D variable representing [files, windows, value],
                                                     where each file has the same number of windows.

    Returns:
        float: The total hours of the dataset represented.
    """
    num_files = len(latent_means_windowed)
    if num_files == 0:
        return 0.0

    num_windows_per_file = len(latent_means_windowed[0])
    if num_windows_per_file == 0:
        return 0.0

    if num_windows_per_file == 1:
        total_seconds_per_file = win_sec
    elif num_windows_per_file > 1:
        # The start time of the first window is 0.
        # The start time of the last window is (num_windows_per_file - 1) * stride_sec.
        # The end time of the last window is (num_windows_per_file - 1) * stride_sec + win_sec.
        total_seconds_per_file = (num_windows_per_file - 1) * stride_sec + win_sec
    else:
        total_seconds_per_file = 0.0

    total_seconds = num_files * total_seconds_per_file
    return total_seconds / 3600.0
        
def format_large_number(n):
    """
    Formats a large number with one decimal point and appropriate suffix (K, M, B, T, Q).

    Args:
        n (float or int): The number to format.

    Returns:
        str: The formatted number with suffix.
    """
    if abs(n) >= 1_000_000_000_000_000:
        return f"{n / 1_000_000_000_000_000:.1f}Q"  # Quadrillion
    elif abs(n) >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"      # Trillion
    elif abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"          # Billion
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"              # Million
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"                  # Thousand
    else:
        return f"{n:.1f}"

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

def plot_hex_grid(ax, data, title, cmap_str='viridis', vmin=None, vmax=None):
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Remove axes

    cmap = cm.get_cmap(cmap_str)
    norm = pl.Normalize(vmin=vmin if vmin is not None else np.min(data), vmax=vmax if vmax is not None else np.max(data))

    # Hexagon geometry
    radius = 1.0  # circumradius
    width = 2 * radius  # not used directly but good to note
    height = np.sqrt(3) * radius  # vertical distance from flat to flat

    rows, cols = data.shape

    for i in range(rows):
        for j in range(cols):
            x = j * 1.5 * radius
            y = i * height + (j % 2) * (height / 2)

            # Scale the raw data to the range [0, 1] based on vmin and vmax
            face_color = cmap(norm(data[i, j]))

            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                            orientation=np.radians(30),
                                            facecolor=face_color, alpha=0.7,
                                            edgecolor=face_color, linewidth=0.1)
            ax.add_patch(hexagon)

    x_extent = cols * 1.5 * radius
    y_extent = rows * height
    ax.set_xlim(-radius, x_extent + radius)
    ax.set_ylim(-radius, y_extent + height)
    ax.set_title(title)

    norm = pl.Normalize(vmin=vmin, vmax=vmax)  # Use vmin and vmax directly
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    pl.colorbar(sm, ax=ax, label=title) # Add a label to the colorbar for clarity

def toroidal_kohonen_subfunction_pytorch(
    atd_file,
    sleep_file,
    skip_sleep,
    pat_ids_list,
    latent_means_windowed,
    latent_logvars_windowed,
    start_datetimes_epoch,
    stop_datetimes_epoch,
    FS,
    win_sec,
    stride_sec,
    savedir,
    subsample_file_factor,
    som_pca_init,
    som_batch_size,
    som_lr,
    som_lr_epoch_decay,
    som_sigma,
    som_sigma_epoch_decay,
    som_epochs,
    som_gridsize,
    som_sigma_min,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    pca_reduce_input = False,
    pca_dims = 50,
    user_cim_kernel_sigma = None,
    som_precomputed_path = None,
    som_object = None,
    som_device = 0,
    sigma_plot=1,
    hits_log_view=True,
    umat_log_view=True,
    preictal_overlay_thresh = 0.25,
    sleep_overlay_thresh = 0.5,
    smooth_map_factor = 1,
    **kwargs):
    """
    Toroidal SOM with hexagonal grid for latent space analysis.
    """

    if not os.path.exists(savedir): os.makedirs(savedir)

    # DATA PREPARATION
    total_dataset_hours = get_dataset_hours(win_sec, stride_sec, latent_means_windowed)
    total_num_embeddings = total_dataset_hours * 3600 * FS
    lr_min = som_lr * (som_lr_epoch_decay ** som_epochs)

    # Metadata
    latent_dim = latent_means_windowed.shape[2]
    num_timepoints_in_windowed_file = latent_means_windowed.shape[1]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(latent_means_windowed.shape[0]):
        if np.sum(np.isnan(latent_means_windowed[i,:,:])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_means_windowed = np.delete(latent_means_windowed, delete_file_idxs, axis=0)
    latent_logvars_windowed = np.delete(latent_logvars_windowed, delete_file_idxs, axis=0)
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Flatten data into [miniepoch, dim] to feed into Kohonen, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_means_input = np.concatenate(latent_means_windowed, axis=0)
    latent_logvars_input = np.concatenate(latent_logvars_windowed, axis=0)
    pat_ids_input = [item for item in pat_ids_list for _ in range(latent_means_windowed[0].shape[0])]
    start_datetimes_input = [item + datetime.timedelta(seconds=stride_sec * i) for item in start_datetimes_epoch for i in range(latent_means_windowed[0].shape[0])]
    stop_datetimes_input = [item + datetime.timedelta(seconds=stride_sec * i) + datetime.timedelta(seconds=win_sec) for item in start_datetimes_epoch for i in range(latent_means_windowed[0].shape[0])]


    # TRAINING

    if som_object is not None:
        print("SOM object passed directly into subfunction, using that as pretrained SOM")
        som = som_object
        som.reset_device(som_device)  # Ensure on proper GPU

    elif som_precomputed_path is not None:
        print(f"Loading Toroidal SOM pretrained weights from FILE: {som_precomputed_path}")
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

        # PCA reduce if indicated
        if pca != None:
            print("Using existing PCA on input data...")
            latent_means_reduced = pca.transform(latent_means_input)
            print(f"PCA complete, new dimensionality of input data is {latent_means_reduced.shape[1]}")
        else: 
            latent_means_reduced = latent_means_input
            print("No PCA applied to input data")

        # Create Toroidal SOM instance with same parameters
        som = ToroidalSOM_2(grid_size=(grid_size, grid_size), input_dim=input_dim, batch_size=batch_size,
                           lr=lr, lr_epoch_decay=lr_epoch_decay, cim_kernel_sigma=cim_kernel_sigma, sigma=sigma,
                           sigma_epoch_decay=sigma_epoch_decay, sigma_min=sigma_min, pca=pca, device=som_device, data_for_init=latent_means_reduced)

        # Load weights
        som.load_state_dict(checkpoint['model_state_dict'])
        som.weights = checkpoint['weights']
        som.reset_device(som_device)

        print(f"Toroidal SOM model loaded from {som_precomputed_path}")

    else:
        # Make new Toroidal SOM and train it

        # PCA reduce if indicated
        if pca_reduce_input and (latent_means_input.shape[1] > pca_dims):
            print("Calculating PCA on input data...")
            pca = PCA(pca_dims, whiten=True)
            latent_means_reduced = pca.fit_transform(latent_means_input)
            print(f"PCA complete, new dimensionality of input data is {pca_dims}")
        else: 
            pca = None
            latent_means_reduced = latent_means_input
            print("No PCA applied to input data")

        # Estimate good CIM sigma & plot histogram of CIM for that sigma
        if user_cim_kernel_sigma is None:
            best_cim, _, _, _ = select_cim_sigma(latent_means_reduced, savename=savedir + f"/GPU{som_device}_EstimatedCIM.jpg")
            print(f"Automating CIM sigma selection, CIM histograms saved and best CIM is estimated to be {best_cim:0.2f}")
        else:
            print(f"User chosen CIM is {user_cim_kernel_sigma:0.2f}")
            best_cim = user_cim_kernel_sigma

        grid_size = (som_gridsize, som_gridsize)
        print(f"Training brand new Toroidal SOM: gridsize:{som_gridsize}, lr:{som_lr} w/ {som_lr_epoch_decay:.4f} decay per epoch to {som_lr * (som_lr_epoch_decay ** som_epochs):.6f}, sigma:{som_sigma} w/ {som_sigma_epoch_decay:.4f} decay per epoch to {som_sigma_min:.6f}")
        som = ToroidalSOM_2(grid_size=grid_size, input_dim=latent_means_reduced.shape[1], batch_size=som_batch_size, lr=som_lr,
                            lr_epoch_decay=som_lr_epoch_decay, cim_kernel_sigma=best_cim, sigma=som_sigma, sigma_epoch_decay=som_sigma_epoch_decay,
                            sigma_min=som_sigma_min, pca=pca, device=som_device, data_for_init=latent_means_reduced)

        # Train and save SOM
        # Even if PCA is being used, still pass in full dims and PCA will be applied after reparam trick gets Z from means & logvars
        som.train(latent_means_input, latent_logvars_input, num_epochs=som_epochs)
        savepath = savedir + f"/GPU{som_device}_ToroidalSOM_ObjectDict_smoothsec{win_sec}_Stride{stride_sec}_subsampleFileFactor{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.6f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_reduced.shape[0]}_dims{latent_means_reduced.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.pt"
        torch.save({
            'model_state_dict': som.state_dict(),
            'weights': som.weights,
            'grid_size': som_gridsize,
            'input_dim': latent_means_reduced.shape[1],
            'pca': pca,
            'lr': som_lr,
            'cim_kernel_sigma': best_cim,
            'sigma': som_sigma,
            'lr_epoch_decay': som_lr_epoch_decay,
            'sigma_epoch_decay': som_sigma_epoch_decay,
            'sigma_min': som_sigma_min,
            'epoch': som_epochs,
            'batch_size': som_batch_size
        }, savepath)
        print(f"Toroidal SOM model saved at {savepath}")

    # PLOT PREPARATION

    # Get preictal weights and sleep stage for each data point
    # Pre-Ictal: 0 = interictal, 0.99999 = immediately before seizure (NOTE: ictal is labeled 0)
    # Sleep: -1 = unlabeled, 0 = wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM
    preictal_float_input, ictal_float_input, sleep_int = preictal_sleep_label(atd_file, sleep_file, skip_sleep, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input)
    print("\nFinished gathering pre-ictal and sleep labels on all data windows")

    # One-hot the sleep data [NA, Wake, N1, N2, N3, REM]
    onehot_sleep = one_hot_encode_with_negatives(sleep_int, -1, 4)

    # Get model weights and coordinates
    weights = som.get_weights()
    hex_coords = som.get_hex_coords()
    grid_size = (som_gridsize, som_gridsize)
    rows, cols = grid_size

    # Initialize maps
    preictal_sums = np.zeros(grid_size)
    ictal_sums = np.zeros(grid_size)
    hit_map = np.zeros(grid_size)
    neuron_patient_dict = {}
    unlabaled_sleep_sums = np.zeros(grid_size)
    wake_sums = np.zeros(grid_size)
    n1_sums = np.zeros(grid_size)
    n2_sums = np.zeros(grid_size)
    n3_sums = np.zeros(grid_size)
    rem_sums = np.zeros(grid_size)

    # SOM Inference on all data in batches
    for i in range(0, len(latent_means_reduced), som_batch_size):

        print(f"Running all data windows through trained Kohonen Map: {i}/{int(len(latent_means_reduced))}                  ", end='\r')

        batch = latent_means_reduced[i:i + som_batch_size]
        batch_patients = pat_ids_input[i:i + som_batch_size]
        batch_preictal_labels = preictal_float_input[i:i + som_batch_size]
        batch_ictal_labels = ictal_float_input[i:i + som_batch_size]
        batch_onhot_sleep_labels = onehot_sleep[i:i + som_batch_size, :]

        batch = torch.tensor(batch, dtype=torch.float32, device=som_device)
        bmu_rows, bmu_cols = som.find_bmu(batch)
        bmu_rows, bmu_cols = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()

        # Update hit map
        np.add.at(hit_map, (bmu_rows, bmu_cols), 1)

        # Process pre-ictal and sleep data
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            # Accumulate preictal scores
            preictal_sums[bmu_row, bmu_col] += batch_preictal_labels[j]
            ictal_sums[bmu_row, bmu_col] += batch_ictal_labels[j]

            # Accumulate sleep labels
            unlabaled_sleep_sums[bmu_row, bmu_col] = batch_onhot_sleep_labels[j, 0]
            wake_sums[bmu_row, bmu_col] += batch_onhot_sleep_labels[j, 1]
            n1_sums[bmu_row, bmu_col] += batch_onhot_sleep_labels[j, 2]
            n2_sums[bmu_row, bmu_col] += batch_onhot_sleep_labels[j, 3]
            n3_sums[bmu_row, bmu_col] += batch_onhot_sleep_labels[j, 4]
            rem_sums[bmu_row, bmu_col] += batch_onhot_sleep_labels[j, 5]

        # Track unique patients per node
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            if (bmu_row, bmu_col) not in neuron_patient_dict:
                neuron_patient_dict[(bmu_row, bmu_col)] = set()
            neuron_patient_dict[(bmu_row, bmu_col)].add(batch_patients[j])

    print("\nFinished Kohonen inference on all data")

    # If hits want to be viewed logarithmically
    if hits_log_view:
        epsilon = np.finfo(float).eps
        hit_map = np.log(hit_map + epsilon)

    # Normalize preictal & ictal sums
    if np.max(preictal_sums) > np.min(preictal_sums):
        preictal_sums = (preictal_sums - np.min(preictal_sums)) / (np.max(preictal_sums) - np.min(preictal_sums))

    if np.max(ictal_sums) > np.min(ictal_sums):
        ictal_sums = (ictal_sums - np.min(ictal_sums)) / (np.max(ictal_sums) - np.min(ictal_sums))

    # Normalize Sleep Sums & Smooth
    if np.max(wake_sums) > np.min(wake_sums):
        wake_sums = gaussian_filter(wake_sums, sigma=smooth_map_factor)
        wake_sums = (wake_sums - np.min(wake_sums)) / (np.max(wake_sums) - np.min(wake_sums))  
    if np.max(n1_sums) > np.min(n1_sums):
        n1_sums = gaussian_filter(n1_sums, sigma=smooth_map_factor)
        n1_sums = (n1_sums - np.min(n1_sums)) / (np.max(n1_sums) - np.min(n1_sums))  
    if np.max(n2_sums) > np.min(n2_sums):
        n2_sums = gaussian_filter(n2_sums, sigma=smooth_map_factor)
        n2_sums = (n2_sums - np.min(n2_sums)) / (np.max(n2_sums) - np.min(n2_sums))  
    if np.max(n3_sums) > np.min(n3_sums):
        n3_sums = gaussian_filter(n3_sums, sigma=smooth_map_factor)
        n3_sums = (n3_sums - np.min(n3_sums)) / (np.max(n3_sums) - np.min(n3_sums))  
    if np.max(rem_sums) > np.min(rem_sums):
        rem_sums = gaussian_filter(rem_sums, sigma=smooth_map_factor)
        rem_sums = (rem_sums - np.min(rem_sums)) / (np.max(rem_sums) - np.min(rem_sums))  

    # Create patient diversity map
    patient_map = np.zeros(grid_size)
    max_unique_patients = max(len(pats) for pats in neuron_patient_dict.values()) if neuron_patient_dict else 1
    for (bmu_row, bmu_col), patients in neuron_patient_dict.items():
        patient_map[bmu_row, bmu_col] = len(patients) / max_unique_patients

    # Compute U-Matrix (using Euclidean distances on toroidal grid) for hexagonal grid
    u_matrix_hex = np.zeros(grid_size)
    for i in range(rows):
        for j in range(cols):
            current_weight = weights[i, j]
            neighbor_distances = []

            # Define hexagonal neighbors with toroidal wrapping
            if i % 2 == 0:
                neighbor_offsets = [(0, 1), (0, -1), (-1, 0), (-1, -1), (1, 0), (1, -1)]
            else:
                neighbor_offsets = [(0, 1), (0, -1), (-1, 1), (-1, 0), (1, 1), (1, 0)]

            for offset_row, offset_col in neighbor_offsets:
                ni = (i + offset_row + rows) % rows
                nj = (j + offset_col + cols) % cols
                neighbor_weight = weights[ni, nj]
                distance = np.linalg.norm(current_weight - neighbor_weight)
                neighbor_distances.append(distance)

            u_matrix_hex[i, j] = np.mean(neighbor_distances) if neighbor_distances else 0

     # If U-Matrix is decided to be viewed logarithmically
    if umat_log_view:
        epsilon = np.finfo(float).eps
        u_matrix_hex = np.log(u_matrix_hex + epsilon)   

    # Apply smoothing
    preictal_sums_smoothed = gaussian_filter(preictal_sums, sigma=smooth_map_factor)
    if np.max(preictal_sums_smoothed) > np.min(preictal_sums_smoothed):
        preictal_sums_smoothed = (preictal_sums_smoothed - np.min(preictal_sums_smoothed)) / (np.max(preictal_sums_smoothed) - np.min(preictal_sums_smoothed))

    # Calculate rescaled map (preictal * patient diversity)
    rescale_preictal = preictal_sums * patient_map
    if np.max(rescale_preictal) > 0:
        rescale_preictal = rescale_preictal / np.max(rescale_preictal)

    # Smooth the rescaled PRE-ICTAL map
    rescale_preictal_smoothed = gaussian_filter(rescale_preictal, sigma=smooth_map_factor)
    if np.max(rescale_preictal_smoothed) > 0:
        rescale_preictal_smoothed = rescale_preictal_smoothed / np.max(rescale_preictal_smoothed)

    # Smooth the ICTAL map
    ictal_smoothed = gaussian_filter(ictal_sums, sigma=smooth_map_factor)
    if np.max(ictal_smoothed) > 0:
        ictal_smoothed = ictal_smoothed / np.max(ictal_smoothed)


    # PLOTTING (2D Hexagonal Plots)

    print("2D Plotting")

    fig_2d, axes_2d = pl.subplots(2, 4, figsize=(28, 12))
    ax_hit = axes_2d[0, 1]
    ax_preictal = axes_2d[0, 2]
    ax_patient = axes_2d[1, 1]
    ax_preictal_smooth = axes_2d[0, 3]
    ax_rescaled = axes_2d[1, 2]
    ax_rescaled_smooth = axes_2d[1, 3]
    ax_umatrix = axes_2d[0, 0]
    ax_comp = axes_2d[1, 0]

    # 1. U-Matrix (Hexagonal, Toroidal)
    plot_hex_grid(ax_umatrix, u_matrix_hex, "U-Matrix (Toroidal, Hexagonal)" + 
                  f"\nTotal Embeddings: {format_large_number(total_num_embeddings)}", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # 2. Hit Map (Hexagonal, Toroidal)
    plot_hex_grid(ax_hit, hit_map, "Hit Map (Hexagonal, Toroidal)", cmap_str='Blues', vmin=0, vmax=np.max(hit_map) if np.max(hit_map) > 0 else 1)

    # 3. Component Plane (Feature 0) (Hexagonal, Toroidal)
    weights_feature_0 = weights[:, :, 0]
    abs_max = np.max(np.abs(weights_feature_0))
    plot_hex_grid(ax_comp, weights_feature_0, "Component Plane (Feature 0, Hexagonal, Toroidal)", cmap_str='coolwarm', vmin=-abs_max, vmax=abs_max)

    # 4. Patient Diversity Map (Hexagonal, Toroidal)
    plot_hex_grid(ax_patient, patient_map, "Patient Diversity Map (1.0 = Most Blended, Hexagonal, Toroidal)", cmap_str='viridis', vmin=0, vmax=1)

    # 5. Pre-Ictal Density (Hexagonal, Toroidal)
    num_wins_preictal = np.count_nonzero(np.array(preictal_float_input) != 0.0)
    if win_sec==stride_sec: approx_num_hours_pre_ictal = num_wins_preictal * win_sec / 3600 # IF Nonoverlapping
    else: approx_num_hours_pre_ictal = -1
    plot_hex_grid(ax_preictal, preictal_sums, f"Pre-Ictal Density (Hexagonal, Toroidal)" + 
                    f"\n{num_wins_preictal}/{len(preictal_float_input)}" + 
                    f"({num_wins_preictal/len(preictal_float_input) * 100:.4f}%) Windows Clinically Pre-Ictal" +
                    f"\n({approx_num_hours_pre_ictal:.1f}/{total_dataset_hours:.1f}) Hours Clinically Pre-Ictal",
                cmap_str='flare', vmin=0, vmax=1)

    # 6. Pre-Ictal * Patient Diversity (Hexagonal, Toroidal)
    plot_hex_grid(ax_rescaled, rescale_preictal, "Pre-Ictal * Patient Diversity (Hexagonal, Toroidal)",
                cmap_str='flare', vmin=0, vmax=1)

    # 7. Pre-Ictal Density (Smoothed) (Hexagonal, Toroidal)
    plot_hex_grid(ax_preictal_smooth, preictal_sums_smoothed, f"Pre-Ictal Density - Smoothed (S:{sigma_plot}, Hexagonal, Toroidal)",
                cmap_str='flare', vmin=0, vmax=1)

    # 8. Pre-Ictal * Patient Diversity (Smoothed) (Hexagonal, Toroidal)
    plot_hex_grid(ax_rescaled_smooth, rescale_preictal_smoothed, f"Pre-Ictal * Patient Diversity - Smoothed (S:{sigma_plot}, Hexagonal, Toroidal)",
                cmap_str='flare', vmin=0, vmax=1)

    # Export 2D figure
    print("Exporting Toroidal SOM 2D visualizations to JPG")
    savename_jpg_2d = savedir + f"/GPU{som_device}_2DPlots_ToroidalSOM.jpg"
    pl.savefig(savename_jpg_2d, dpi=600)
    pl.savefig(savename_jpg_2d.replace('.jpg', '.svg'))


    # 2D OVERLAY: U-Matrix + Pre-Ictal
    
    # Create new figure for U-Matrix + Pre-Ictal Density Overlay
    fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

    # Clip preictal_sums_smoothed at lower threshold 0.5
    overlay_preictal = np.clip(preictal_sums_smoothed, 0.0, 1.0)
    # overlay_preictal = np.clip(rescale_preictal_smoothed, 0.0, 1.0)
    # overlay_preictal = np.clip(preictal_sums, 0.0, 1.0)

    # Plot U-Matrix base
    plot_hex_grid(ax_overlay, u_matrix_hex, "U-Matrix with Pre-Ictal Overlay", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # Overlay Pre-Ictal smoothed density (alpha=0.6)
    # We'll replot on top with a semi-transparent flare colormap
    rows, cols = overlay_preictal.shape
    radius = 1.0
    height = np.sqrt(3) * radius
    cmap_overlay = cm.get_cmap('flare')
    norm_overlay = pl.Normalize(vmin=0.0, vmax=1.0)

    for i in range(rows):
        for j in range(cols):
            x = j * 1.5 * radius
            y = i * height + (j % 2) * (height / 2)
            face_color = cmap_overlay(norm_overlay(overlay_preictal[i, j]))
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                            orientation=np.radians(30),
                                            facecolor=face_color, alpha=0.7,
                                            edgecolor=None, linewidth=0)
            if overlay_preictal[i, j] >= preictal_overlay_thresh:
                ax_overlay.add_patch(hexagon)

    # Optional: add a colorbar for the overlay
    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    pl.colorbar(sm_overlay, ax=ax_overlay, label="Pre-Ictal Density (Clipped & Smoothed)")

    # Save overlay figure
    savename_overlay = savedir + f"/GPU{som_device}_UMatrix_PreIctalOverlay_ToroidalSOM.jpg"
    pl.savefig(savename_overlay, dpi=600)
    pl.savefig(savename_overlay.replace('.jpg', '.svg'))
    
    pickle_path = savedir + "/overlay_figure_object.pkl"
    output_obj = open(pickle_path, 'wb')
    pickle.dump({
        'fig':fig_overlay, 
        'ax': ax_overlay,
        'u_matrix_hex': u_matrix_hex,
        'preictal_sums': preictal_sums,
        'preictal_sums_smoothed': preictal_sums_smoothed,
        'rescale_preictal_smoothed': rescale_preictal_smoothed
        }, 
        output_obj)
    output_obj.close()
    print("Saved overlay figure objects for later use")


    # 2D OVERLAY: U-Matrix + ICTAL
    
    # Create new figure for U-Matrix + Pre-Ictal Density Overlay
    fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

    # Pull ictal data
    # overlay_ictal = np.clip(ictal_sums, 0.0, 1.0)
    overlay_ictal = np.clip(ictal_smoothed, 0.0, 1.0)

    # Plot U-Matrix base
    plot_hex_grid(ax_overlay, u_matrix_hex, "U-Matrix with ICTAL Overlay", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # Overlay Pre-Ictal smoothed density (alpha=0.6)
    # We'll replot on top with a semi-transparent flare colormap
    rows, cols = overlay_ictal.shape
    radius = 1.0
    height = np.sqrt(3) * radius
    cmap_overlay = cm.get_cmap('Purples')
    norm_overlay = pl.Normalize(vmin=0.0, vmax=1.0)

    for i in range(rows):
        for j in range(cols):
            x = j * 1.5 * radius
            y = i * height + (j % 2) * (height / 2)
            face_color = cmap_overlay(norm_overlay(overlay_ictal[i, j]))
            hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                            orientation=np.radians(30),
                                            facecolor=face_color, alpha=0.7,
                                            edgecolor=None, linewidth=0)
            if overlay_ictal[i, j] >= preictal_overlay_thresh:
                ax_overlay.add_patch(hexagon)

    # Optional: add a colorbar for the overlay
    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    pl.colorbar(sm_overlay, ax=ax_overlay, label="ICTAL Density")

    # Save overlay figure
    savename_overlay = savedir + f"/GPU{som_device}_UMatrix_ICTAL_Overlay_ToroidalSOM.jpg"
    pl.savefig(savename_overlay, dpi=600)
    pl.savefig(savename_overlay.replace('.jpg', '.svg'))


    # 2D OVERLAY: U-Matrix + SLEEP single plots 
    if not skip_sleep:
        print("Plotting Sleep Data: Single Plots")
        sleep_strings = ['Wake', 'N1', 'N2', 'N3', 'REM']
        sleep_color_strings = ['Purples','Purples','Purples','Purples','Purples']
        sleep_concat = [wake_sums, n1_sums, n2_sums, n3_sums, rem_sums]
        for s_idx in range(0,5): # Wake, N1, N2, N3, REM
            # Create new figure for U-Matrix + Pre-Ictal Density Overlay
            fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

            # Pull ictal data
            # overlay_ictal = np.clip(ictal_sums, 0.0, 1.0)
            color_str = sleep_color_strings[s_idx]
            sleep_string_curr = sleep_strings[s_idx]
            overlay_data = np.clip(sleep_concat[s_idx], 0.0, 1.0)

            # Plot U-Matrix base
            plot_hex_grid(ax_overlay, u_matrix_hex, f"U-Matrix with {sleep_string_curr} Overlay", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

            # Overlay Pre-Ictal smoothed density (alpha=0.6)
            # We'll replot on top with a semi-transparent flare colormap
            rows, cols = overlay_data.shape
            radius = 1.0
            height = np.sqrt(3) * radius
            cmap_overlay = cm.get_cmap(color_str)
            norm_overlay = pl.Normalize(vmin=0.0, vmax=1.0)

            for i in range(rows):
                for j in range(cols):
                    x = j * 1.5 * radius
                    y = i * height + (j % 2) * (height / 2)
                    face_color = cmap_overlay(norm_overlay(overlay_data[i, j]))
                    hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius,
                                                    orientation=np.radians(30),
                                                    facecolor=face_color, alpha=0.7,
                                                    edgecolor=None, linewidth=0)
                    if overlay_data[i, j] >= sleep_overlay_thresh:
                        ax_overlay.add_patch(hexagon)

            # Optional: add a colorbar for the overlay
            sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
            sm_overlay.set_array([])
            pl.colorbar(sm_overlay, ax=ax_overlay, label=f"{sleep_string_curr} Density")

            # Save overlay figure
            savename_overlay = savedir + f"/GPU{som_device}_UMatrix_{sleep_string_curr}_Overlay_ToroidalSOM.jpg"
            pl.savefig(savename_overlay, dpi=600)
            pl.savefig(savename_overlay.replace('.jpg', '.svg'))

    # 2D OVERLAY: U-Matrix + All Sleep Stages On One
    if not skip_sleep:
        print("Plotting Sleep Data: Combined Plot")
        # sleep_strings = ['Wake', 'N1', 'N2', 'N3', 'REM']
        # sleep_colors = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        # sleep_concat = [wake_sums, n1_sums, n2_sums, n3_sums, rem_sums]
        sleep_strings = ['N2', 'N3', 'REM']
        sleep_colors = ['Blues', 'Purples', 'Greens']
        sleep_concat = [n2_sums, n3_sums, rem_sums]

        # Create a single figure for U-Matrix + All Sleep Stages Overlay
        fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

        # Plot U-Matrix base
        plot_hex_grid(
            ax_overlay, u_matrix_hex,
            "U-Matrix with All Sleep Stages Overlay",
            cmap_str='bone_r',
            vmin=np.min(u_matrix_hex),
            vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1
        )

        rows, cols = u_matrix_hex.shape
        radius = 1.0
        height = np.sqrt(3) * radius

        for s_idx in range(len(sleep_strings)):  
            overlay_data = np.clip(sleep_concat[s_idx], 0.0, 1.0)
            cmap_overlay = cm.get_cmap(sleep_colors[s_idx])
            norm_overlay = pl.Normalize(vmin=0.0, vmax=1.0)

            for i in range(rows):
                for j in range(cols):
                    x = j * 1.5 * radius
                    y = i * height + (j % 2) * (height / 2)
                    value = overlay_data[i, j]
                    if value >= sleep_overlay_thresh:
                        face_color = cmap_overlay(norm_overlay(value))
                        hexagon = patches.RegularPolygon(
                            (x, y), numVertices=6, radius=radius,
                            orientation=np.radians(30),
                            facecolor=face_color, alpha=0.5,
                            edgecolor=None, linewidth=0
                        )
                        ax_overlay.add_patch(hexagon)

        # Save overlay figure
        savename_overlay = savedir + f"/GPU{som_device}_UMatrix_AllSleepStages_Overlay_ToroidalSOM.jpg"
        pl.savefig(savename_overlay, dpi=600)



    # PLOTTING (All Plots in 3D - Representing Toroidal as a Flat Grid)

    fig_3d = pl.figure(figsize=(28, 24))

    # Function to create a 3D plot of the 2D grid
    def plot_3d_grid(fig, ax, data, title, cmap_str='viridis', vmin=None, vmax=None):
        rows, cols = data.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        Z = data  # Use the data directly as the Z-height

        cmap = cm.get_cmap(cmap_str)
        norm = pl.Normalize(vmin=vmin if vmin is not None else np.min(data), vmax=vmax if vmax is not None else np.max(data))
        colors = cmap(norm(Z))

        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=0.7)
        ax.set_xlabel("SOM Column")
        ax.set_ylabel("SOM Row")
        ax.set_zlabel("Value")
        ax.set_title(title)

        sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=title, shrink=0.6)

    # 1. U-Matrix (3D Grid)
    ax_umatrix_3d = fig_3d.add_subplot(2, 4, 1, projection='3d')
    plot_3d_grid(fig_3d, ax_umatrix_3d, u_matrix_hex, "U-Matrix (Toroidal)", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # 2. Hit Map (3D Grid)
    ax_hit_3d = fig_3d.add_subplot(2, 4, 2, projection='3d')
    plot_3d_grid(fig_3d, ax_hit_3d, hit_map, "Hit Map (Toroidal)", cmap_str='Blues', vmin=0, vmax=np.max(hit_map) if np.max(hit_map) > 0 else 1)

    # 3. Component Plane (Feature 0) (3D Grid)
    weights_feature_0 = weights[:, :, 0]
    abs_max = np.max(np.abs(weights_feature_0))
    ax_comp_3d = fig_3d.add_subplot(2, 4, 3, projection='3d')
    plot_3d_grid(fig_3d, ax_comp_3d, weights_feature_0, "Component Plane (Feature 0, Toroidal)", cmap_str='coolwarm', vmin=-abs_max, vmax=abs_max)

    # 4. Patient Diversity Map (3D Grid)
    ax_patient_3d = fig_3d.add_subplot(2, 4, 4, projection='3d')
    plot_3d_grid(fig_3d, ax_patient_3d, patient_map, "Patient Diversity (Toroidal)", cmap_str='viridis', vmin=0, vmax=1)

    # 5. Pre-Ictal Density (3D Grid)
    ax_preictal_3d = fig_3d.add_subplot(2, 4, 5, projection='3d')
    plot_3d_grid(fig_3d, ax_preictal_3d, preictal_sums, "Pre-Ictal Density (Toroidal)", cmap_str='flare', vmin=0, vmax=1)

    # 6. Pre-Ictal * Patient Diversity (3D Grid)
    ax_rescaled_3d = fig_3d.add_subplot(2, 4, 6, projection='3d')
    plot_3d_grid(fig_3d, ax_rescaled_3d, rescale_preictal, "Pre-Ictal * Patient Diversity (Toroidal)", cmap_str='flare', vmin=0, vmax=1)

    # 7. Pre-Ictal Density (Smoothed) (3D Grid)
    ax_preictal_smooth_3d = fig_3d.add_subplot(2, 4, 7, projection='3d')
    plot_3d_grid(fig_3d, ax_preictal_smooth_3d, preictal_sums_smoothed, f"Pre-Ictal Density - Smoothed (S:{sigma_plot}, Toroidal)", cmap_str='flare', vmin=0, vmax=1)

    # 8. Pre-Ictal * Patient Diversity (Smoothed) (3D Grid)
    ax_rescaled_smooth_3d = fig_3d.add_subplot(2, 4, 8, projection='3d')

    ax_rescaled_smooth_3d = fig_3d.add_subplot(2, 4, 8, projection='3d')
    plot_3d_grid(fig_3d, ax_rescaled_smooth_3d, rescale_preictal_smoothed, f"Pre-Ictal * Patient Diversity - Smoothed (S:{sigma_plot}, Toroidal)",
                cmap_str='flare', vmin=0, vmax=1)

    # Export 3D figure
    print("Exporting Toroidal SOM 3D visualizations to JPG")
    savename_jpg_3d = savedir + f"/GPU{som_device}_3DPlots.jpg"
    pl.savefig(savename_jpg_3d, dpi=600)
    pl.savefig(savename_jpg_3d.replace('.jpg', '.svg'))


    # PLOTTING (All Plots in 3D - Projected onto a Toroid)

    fig_toroid = pl.figure(figsize=(28, 24))

    def plot_3d_on_toroid(fig, ax, data, title, cmap_str='viridis', vmin=None, vmax=None):
        rows, cols = data.shape
        u, v = np.mgrid[0:2*np.pi:cols*1j, 0:2*np.pi:rows*1j]
        r = 1
        R = 2
        x = (R + r*np.cos(v)) * np.cos(u)
        y = (R + r*np.cos(v)) * np.sin(u)
        z = r * np.sin(v)

        cmap = cm.get_cmap(cmap_str)
        norm = pl.Normalize(vmin=vmin if vmin is not None else np.min(data), vmax=vmax if vmax is not None else np.max(data))
        colors = cmap(norm(data.T)) # Transpose data to match u,v grid

        ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, alpha=0.7)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.set_axis_off()

        sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=title, shrink=0.6)

    # 1. U-Matrix (Toroid)
    ax_umatrix_toroid = fig_toroid.add_subplot(2, 4, 1, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_umatrix_toroid, u_matrix_hex, "U-Matrix (Toroid)", cmap_str='bone_r', vmin=np.min(u_matrix_hex), vmax=np.max(u_matrix_hex) if np.max(u_matrix_hex) > 0 else 1)

    # 2. Hit Map (Toroid)
    ax_hit_toroid = fig_toroid.add_subplot(2, 4, 2, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_hit_toroid, hit_map, "Hit Map (Toroid)", cmap_str='Blues', vmin=0, vmax=np.max(hit_map) if np.max(hit_map) > 0 else 1)

    # 3. Component Plane (Feature 0) (Toroid)
    weights_feature_0 = weights[:, :, 0]
    abs_max = np.max(np.abs(weights_feature_0))
    ax_comp_toroid = fig_toroid.add_subplot(2, 4, 3, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_comp_toroid, weights_feature_0, "Component Plane (Feature 0, Toroid)", cmap_str='coolwarm', vmin=-abs_max, vmax=abs_max)

    # 4. Patient Diversity Map (Toroid)
    ax_patient_toroid = fig_toroid.add_subplot(2, 4, 4, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_patient_toroid, patient_map, "Patient Diversity (Toroid)", cmap_str='viridis', vmin=0, vmax=1)

    # 5. Pre-Ictal Density (Toroid)
    ax_preictal_toroid = fig_toroid.add_subplot(2, 4, 5, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_preictal_toroid, preictal_sums, "Pre-Ictal Density (Toroid)", cmap_str='flare', vmin=0, vmax=1)

    # 6. Pre-Ictal * Patient Diversity (Toroid)
    ax_rescaled_toroid = fig_toroid.add_subplot(2, 4, 6, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_rescaled_toroid, rescale_preictal, "Pre-Ictal * Patient Diversity (Toroid)", cmap_str='flare', vmin=0, vmax=1)

    # 7. Pre-Ictal Density (Smoothed) (Toroid)
    ax_preictal_smooth_toroid = fig_toroid.add_subplot(2, 4, 7, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_preictal_smooth_toroid, preictal_sums_smoothed, f"Pre-Ictal Density - Smoothed (S:{sigma_plot}, Toroid)", cmap_str='flare', vmin=0, vmax=1)

    # 8. Pre-Ictal * Patient Diversity (Smoothed) (Toroid)
    ax_rescaled_smooth_toroid = fig_toroid.add_subplot(2, 4, 8, projection='3d')
    plot_3d_on_toroid(fig_toroid, ax_rescaled_smooth_toroid, rescale_preictal_smoothed, f"Pre-Ictal * Patient Diversity - Smoothed (S:{sigma_plot}, Toroid)",
                        cmap_str='flare', vmin=0, vmax=1)

    # Export Toroid figure
    print("Exporting Toroidal SOM visualizations projected onto a Toroid to JPG")
    savename_jpg_toroid = savedir + f"/GPU{som_device}_ToroidalSOM.jpg"
    pl.savefig(savename_jpg_toroid, dpi=600)
    pl.savefig(savename_jpg_toroid.replace('.jpg', '.svg'))

def plot_mog_histograms(
    savedir, means_full, mogpreds_full,
    ax_lims, max_tier1_chunks,
    batch_idx=0, big_window_indices=None,
    num_dims=3,
    window_size=None, stride=None
    
):
    """
    Plot MoG component-wise histograms (tier0) and abstracted average histograms (tier1) over time.

    Parameters:
        savedir: str, directory to save plots
        means_full: np.ndarray, shape (B, W, S, K, D)
        mogpreds_full: np.ndarray, shape (B, W, S, K)
        batch_idx: int
        big_window_indices: list of int, which big_window indices to include
        ax_lims: tuple, (xmin, xmax) for x-axis scaling on histograms
        num_dims: int, number of latent dimensions to plot
        window_size: int or None, how many windows to average over for tier1
        stride: int or None, stride for moving window in tier1
    """
    import numpy as np
    import matplotlib.pyplot as pl
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if big_window_indices is None:
        big_window_indices = [0]

    num_windows = len(big_window_indices)
    num_components = means_full.shape[3]
    max_dims = means_full.shape[-1]
    num_dims = min(num_dims, max_dims)

    # -- TIER 0A PLOTS (Show All BSE encode_samples) -- #
    base_cmap = cm.get_cmap('bone')
    clip_upper = 0.85
    norm = Normalize(vmin=min(big_window_indices), vmax=max(big_window_indices))
    window_colors = [base_cmap(clip_upper * norm(idx)) for idx in big_window_indices]

    fig, axes = pl.subplots(nrows=num_components, ncols=1 + num_dims, figsize=(4 * (1 + num_dims), 3 * num_components))
    fig2, weighted_axes = pl.subplots(nrows=1, ncols=num_dims, figsize=(4 * num_dims, 3))

    if num_components == 1:
        axes = np.expand_dims(axes, axis=0)

    for comp in range(num_components):
        # COLUMN 0: Mean MoG weights
        ax0 = axes[comp, 0]
        means = [mogpreds_full[batch_idx, win, :, comp].mean() for win in big_window_indices]
        ax0.bar(range(num_windows), means, color=window_colors, width=0.8)
        ax0.set_ylim([0, 1])
        ax0.set_xticks(range(num_windows))
        ax0.set_xticklabels([f"W{i}" for i in big_window_indices])
        ax0.set_title(f'Mean Pred C{comp}')
        ax0.set_ylabel('Mean Weight')

        # Columns 1+ : per-dim histograms
        for dim in range(num_dims):
            ax = axes[comp, dim + 1]
            for i, win_idx in enumerate(big_window_indices):
                data = means_full[batch_idx, win_idx, :, comp, dim]
                ax.hist(data, bins=20, alpha=0.5, label=f"W{win_idx}", color=window_colors[i])
            ax.set_xlim(ax_lims)
            ax.set_title(f'Latent {dim}')
            # ax.grid(True)
            # if comp == 0 and dim == num_dims - 1:
            #     ax.legend(loc='upper right', fontsize='small')

    for dim in range(num_dims):
        ax = weighted_axes[dim]
        for i, win_idx in enumerate(big_window_indices):
            weighted_vals = (
                mogpreds_full[batch_idx, win_idx, :, :, np.newaxis] *
                means_full[batch_idx, win_idx, :, :, dim:dim+1]
            ).sum(axis=1).flatten()
            ax.hist(weighted_vals, bins=20, alpha=0.6, label=f"W{win_idx}", color=window_colors[i])
        ax.set_xlim(ax_lims)
        ax.set_title(f'Weighted Mean (Dim {dim})')
        # ax.grid(True)
        # if dim == num_dims - 1:
        #     ax.legend(loc='upper right', fontsize='small')

    fig.savefig(f"{savedir}/tier0A_components.jpg")
    fig2.savefig(f"{savedir}/tier0A_weighted_means.jpg")


    # -- TIER 0B PLOTS (Averaged across BSE encode_samples) -- #
    base_cmap = cm.get_cmap('bone')
    clip_upper = 0.85
    norm = Normalize(vmin=min(big_window_indices), vmax=max(big_window_indices))
    window_colors = [base_cmap(clip_upper * norm(idx)) for idx in big_window_indices]

    fig, axes = pl.subplots(nrows=num_components, ncols=1 + num_dims, figsize=(4 * (1 + num_dims), 3 * num_components))
    fig2, weighted_axes = pl.subplots(nrows=1, ncols=num_dims, figsize=(4 * num_dims, 3))

    if num_components == 1:
        axes = np.expand_dims(axes, axis=0)

    for comp in range(num_components):
        # COLUMN 0: Mean MoG weights
        ax0 = axes[comp, 0]
        means = [mogpreds_full[batch_idx, win, :, comp].mean() for win in big_window_indices]
        ax0.bar(range(num_windows), means, color=window_colors, width=0.8)
        ax0.set_ylim([0, 1])
        ax0.set_xticks(range(num_windows))
        ax0.set_xticklabels([f"W{i}" for i in big_window_indices])
        ax0.set_title(f'Mean Pred C{comp}')
        ax0.set_ylabel('Mean Weight')

        # Columns 1+ : per-dim histograms
        for dim in range(num_dims):
            ax = axes[comp, dim + 1]
            for i, win_idx in enumerate(big_window_indices):
                data = np.mean(means_full[batch_idx, win_idx, :, comp, dim])
                ax.hist(data, bins=20, alpha=0.5, label=f"W{win_idx}", color=window_colors[i])
            ax.set_xlim(ax_lims)
            ax.set_title(f'Latent {dim}')
            # ax.grid(True)
            # if comp == 0 and dim == num_dims - 1:
            #     ax.legend(loc='upper right', fontsize='small')

    for dim in range(num_dims):
        ax = weighted_axes[dim]
        for i, win_idx in enumerate(big_window_indices):
            weighted_vals = (
                mogpreds_full[batch_idx, win_idx, :, :, np.newaxis] *
                means_full[batch_idx, win_idx, :, :, dim:dim+1]
            ).sum(axis=1).flatten().mean()
            ax.hist(weighted_vals, bins=20, alpha=0.6, label=f"W{win_idx}", color=window_colors[i])
        ax.set_xlim(ax_lims)
        ax.set_title(f'Weighted Mean (Dim {dim})')
        # ax.grid(True)
        # if dim == num_dims - 1:
        #     ax.legend(loc='upper right', fontsize='small')

    fig.savefig(f"{savedir}/tier0B_components.jpg")
    fig2.savefig(f"{savedir}/tier0B_weighted_means.jpg")

    # -- TIER 1A PLOTS (Temporal Abstraction, but show all BSE encode_samples ) -- #
    if window_size is not None and stride is not None:
        tier1_chunks = []
        chunk_labels = []
        idx_array = np.arange(means_full.shape[1])
        for i in range(0, len(idx_array) - window_size + 1, stride):
            if max_tier1_chunks is not None and len(tier1_chunks) >= max_tier1_chunks:
                break
            chunk = idx_array[i:i + window_size]
            tier1_chunks.append(chunk)
            chunk_labels.append(f"{chunk[0]}-{chunk[-1]}")

        n_chunks = len(tier1_chunks)
        tier1_colors = [base_cmap(clip_upper * i / max(1, n_chunks - 1)) for i in range(n_chunks)]

        fig3, axes3 = pl.subplots(nrows=num_components, ncols=1 + num_dims, figsize=(4 * (1 + num_dims), 3 * num_components))
        fig4, weighted_axes2 = pl.subplots(nrows=1, ncols=num_dims, figsize=(4 * num_dims, 3))
        if num_components == 1:
            axes3 = np.expand_dims(axes3, axis=0)

        for comp in range(num_components):
            # COLUMN 0: Averaged MoG weights
            ax0 = axes3[comp, 0]
            mean_vals = []
            for chunk in tier1_chunks:
                vals = mogpreds_full[batch_idx, chunk, :, comp]
                mean_vals.append(vals.mean())
            ax0.bar(range(n_chunks), mean_vals, color=tier1_colors, width=0.8)
            ax0.set_ylim([0, 1])
            ax0.set_xticks(range(n_chunks))
            ax0.set_xticklabels(chunk_labels, rotation=45)
            ax0.set_title(f'Tier1A Mean Pred C{comp}')

            for dim in range(num_dims):
                ax = axes3[comp, dim + 1]
                for i, chunk in enumerate(tier1_chunks):
                    data = means_full[batch_idx, chunk, :, comp, dim].reshape(-1)
                    ax.hist(data, bins=20, alpha=0.5, label=chunk_labels[i], color=tier1_colors[i])
                ax.set_xlim(ax_lims)
                ax.set_title(f'Latent {dim}')
                # ax.grid(True)
                # if comp == 0 and dim == num_dims - 1:
                #     ax.legend(loc='upper right', fontsize='small')

        for dim in range(num_dims):
            ax = weighted_axes2[dim]
            for i, chunk in enumerate(tier1_chunks):
                preds = mogpreds_full[batch_idx, chunk, :, :, np.newaxis]  # (W, S, K, 1)
                means = means_full[batch_idx, chunk, :, :, dim:dim+1]      # (W, S, K, 1)
                weighted_vals = (preds * means).sum(axis=2).reshape(-1)    # sum over components, flatten over W and S
                ax.hist(weighted_vals, bins=20, alpha=0.6, label=chunk_labels[i], color=tier1_colors[i])
            ax.set_xlim(ax_lims)
            ax.set_title(f'Tier1A Weighted Mean (Dim {dim})')
            # ax.grid(True)
            # if dim == num_dims - 1:
            #     ax.legend(loc='upper right', fontsize='small')

        fig3.savefig(f"{savedir}/tier1A_components.jpg")
        fig4.savefig(f"{savedir}/tier1A_weighted_means.jpg")

    # -- TIER 1B PLOTS (Temporal Abstraction, averaged across BSE encode_samples) -- #
    if window_size is not None and stride is not None:
        tier1_chunks = []
        chunk_labels = []
        idx_array = np.arange(means_full.shape[1])
        for i in range(0, len(idx_array) - window_size + 1, stride):
            if max_tier1_chunks is not None and len(tier1_chunks) >= max_tier1_chunks:
                break
            chunk = idx_array[i:i + window_size]
            tier1_chunks.append(chunk)
            chunk_labels.append(f"{chunk[0]}-{chunk[-1]}")

        n_chunks = len(tier1_chunks)
        tier1_colors = [base_cmap(clip_upper * i / max(1, n_chunks - 1)) for i in range(n_chunks)]

        fig3, axes3 = pl.subplots(nrows=num_components, ncols=1 + num_dims, figsize=(4 * (1 + num_dims), 3 * num_components))
        fig4, weighted_axes2 = pl.subplots(nrows=1, ncols=num_dims, figsize=(4 * num_dims, 3))
        if num_components == 1:
            axes3 = np.expand_dims(axes3, axis=0)

        for comp in range(num_components):
            # COLUMN 0: Averaged MoG weights
            ax0 = axes3[comp, 0]
            mean_vals = []
            for chunk in tier1_chunks:
                vals = mogpreds_full[batch_idx, chunk, :, comp]
                mean_vals.append(vals.mean())
            ax0.bar(range(n_chunks), mean_vals, color=tier1_colors, width=0.8)
            ax0.set_ylim([0, 1])
            ax0.set_xticks(range(n_chunks))
            ax0.set_xticklabels(chunk_labels, rotation=45)
            ax0.set_title(f'Tier1B Mean Pred C{comp}')

            for dim in range(num_dims):
                ax = axes3[comp, dim + 1]
                for i, chunk in enumerate(tier1_chunks):
                    data = means_full[batch_idx, chunk, :, comp, dim].reshape(-1).mean()
                    ax.hist(data, bins=20, alpha=0.5, label=chunk_labels[i], color=tier1_colors[i])
                ax.set_xlim(ax_lims)
                ax.set_title(f'Latent {dim}')
                # ax.grid(True)
                # if comp == 0 and dim == num_dims - 1:
                #     ax.legend(loc='upper right', fontsize='small')

        for dim in range(num_dims):
            ax = weighted_axes2[dim]
            for i, chunk in enumerate(tier1_chunks):
                preds = mogpreds_full[batch_idx, chunk, :, :, np.newaxis]  # (W, S, K, 1)
                means = means_full[batch_idx, chunk, :, :, dim:dim+1]      # (W, S, K, 1)
                weighted_vals = (preds * means).sum(axis=2).reshape(-1).mean()    # sum over components, flatten over W and S
                ax.hist(weighted_vals, bins=20, alpha=0.6, label=chunk_labels[i], color=tier1_colors[i])
            ax.set_xlim(ax_lims)
            ax.set_title(f'Tier1B Weighted Mean (Dim {dim})')
            # ax.grid(True)
            # if dim == num_dims - 1:
            #     ax.legend(loc='upper right', fontsize='small')

        fig3.savefig(f"{savedir}/tier1B_components.jpg")
        fig4.savefig(f"{savedir}/tier1B_weighted_means.jpg")

def compute_histograms(data, min_val, max_val, B):
    """
    Compute histogram bin counts for each dimension of a 2D array.

    Parameters:
        data (np.ndarray): Input 2D array of shape (N, M).
        min_val (float): Minimum value for the histogram range.
        max_val (float): Maximum value for the histogram range.
        B (int): Number of bins for the histogram.

    Returns:
        np.ndarray: A 2D array of shape (M, B) containing bin counts for each dimension.
    """
    # Initialize an array to store the histogram bin counts for each dimension
    histograms = np.zeros((data.shape[1], B), dtype=int)

    # Compute the bin edges
    bin_edges = np.linspace(min_val, max_val, B + 1)

    # Iterate over each dimension (column) of the input data
    for i in range(data.shape[1]):
        # Compute the histogram for the current dimension
        hist, _ = np.histogram(data[:, i], bins=bin_edges)
        histograms[i, :] = hist

    return histograms

# def histogram_latent(
#     pat_ids_list,
#     latent_data_windowed, 
#     start_datetimes_epoch,  
#     stop_datetimes_epoch,
#     epoch, 
#     FS, 
#     win_sec, 
#     stride_sec, 
#     savedir,
#     bincount_perdim=200,
#     perc_max=99.99,
#     perc_min=0.01):

#     # Establish edge of histo bins
#     thresh_max = 0
#     thresh_min = 0
#     for i in range(len(latent_data_windowed)):
#         if np.percentile(latent_data_windowed[i], perc_max) > thresh_max: thresh_max = np.percentile(latent_data_windowed[i],perc_max)
#         if np.percentile(latent_data_windowed[i], perc_min) < thresh_min: thresh_min = np.percentile(latent_data_windowed[i], perc_min)

#     # Range stats
#     thresh_range = thresh_max - thresh_min
#     thresh_step = thresh_range / bincount_perdim
#     all_bin_values = [np.round(thresh_min + i*thresh_step, 2) for i in range(bincount_perdim)]
#     zero_bin = np.argmax(np.array(all_bin_values) > 0)

#     num_ticks = 10 # Manually set 10 x-ticks for bins
#     xtick_positions = np.linspace(0, bincount_perdim - 1, num_ticks).astype(int)  # Create 10 evenly spaced positions
#     xtick_labels = [np.round(thresh_min + i*thresh_step, 2) for i in xtick_positions]  # Create labels for these positions
    
#     # Count up histo for each dim
#     histo_counts = np.zeros([latent_data_windowed[0].shape[1], bincount_perdim], dtype=int)
#     for i in range(len(latent_data_windowed)):
#         out = compute_histograms(latent_data_windowed[i], thresh_min, thresh_max, bincount_perdim)
#         histo_counts = histo_counts + out

#     # Log hist data
#     log_hist_data = np.log1p(histo_counts) 

#     fig = pl.figure(figsize=(10, 25))
#     gs = gridspec.GridSpec(2, 2, figure=fig)

#     # Create a custom colormap from pink to purple
#     white_to_darkpurple_cmap = LinearSegmentedColormap.from_list(
#         "white_to_darkpurple", ["white", "#2E004F"]  # White to dark purple (#2E004F)
#     )

#     sns.heatmap(log_hist_data, cmap=white_to_darkpurple_cmap, cbar=True, yticklabels=False, xticklabels=False)
#     pl.axvline(x=zero_bin, color='gray', linestyle='-', linewidth=2)  # Gray solid line at x = 0

#     pl.xticks(xtick_positions, xtick_labels, rotation=45)

#     # Customize the plot
#     pl.title('Heatmap of Histograms for all Dimensions', fontsize=16)
#     pl.ylabel('Dimensions', fontsize=14)
#     pl.xlabel('Bins', fontsize=14)

#     # **** Save entire figure *****
#     if not os.path.exists(savedir + '/JPEGs'): os.makedirs(savedir + '/JPEGs')
#     savename_jpg = savedir + f"/JPEGs/pacmap_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_epoch" + str(epoch) + f"_bincount{bincount_perdim}.jpg"
#     pl.savefig(savename_jpg, dpi=300)


#     pl.close(fig)

def filename_to_datetimes(list_file_names):
        start_datetimes = [datetime.datetime.min]*len(list_file_names)
        stop_datetimes = [datetime.datetime.min]*len(list_file_names)
        for i in range(0, len(list_file_names)):
            splits = list_file_names[i].split('_')
            aD = splits[1]
            aT = splits[2]
            start_datetimes[i] = datetime.datetime(int(aD[4:8]), int(aD[0:2]), int(aD[2:4]), int(aT[0:2]), int(aT[2:4]), int(aT[4:6]), int(int(aT[6:8])*1e4))
            bD = splits[4]
            bT = splits[5]
            stop_datetimes[i] = datetime.datetime(int(bD[4:8]), int(bD[0:2]), int(bD[2:4]), int(bT[0:2]), int(bT[2:4]), int(bT[4:6]), int(int(bT[6:8])*1e4))
        return start_datetimes, stop_datetimes

def get_pat_seiz_datetimes(
    pat_id, 
    atd_file, 
    FBTC_bool=True, 
    FIAS_bool=True, 
    FAS_to_FIAS_bool=True,
    FAS_bool=True, 
    subclinical_bool=False, 
    focal_unknown_bool=True,
    unknown_bool=True, 
    non_electro_bool=False,
    artifact_bool=False,
    stim_fas_bool=False,
    stim_fias_bool=False,
    **kwargs
    ):

    failure_count = 0
    for i in range(5):
        try:
            atd_df = pd.read_csv(atd_file, sep=',', header='infer')
            pat_seizure_bool = (atd_df['Pat ID'] == pat_id) & (atd_df['Type'] == "Seizure")
            pat_seizurebool_AND_desiredTypes = pat_seizure_bool
            
            # Look for each seizure type individually & delete if not desired
            # seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal, unknown awareness', 'Unknown', 'Non-electrographic']
            seiz_type_list = ['FBTC', 'FIAS', 'FAS_to_FIAS', 'FAS', 'Subclinical', 'Focal unknown awareness', 'Unknown', 'Non-electrographic', 'Artifact', 'Stim-FAS', 'Stim-FIAS']
            delete_seiz_type_bool_list = [FBTC_bool, FIAS_bool, FAS_to_FIAS_bool, FAS_bool, subclinical_bool, focal_unknown_bool, unknown_bool, non_electro_bool, artifact_bool, stim_fas_bool, stim_fias_bool]
            for i in range(0,len(seiz_type_list)):
                if delete_seiz_type_bool_list[i]==False:
                    find_str = seiz_type_list[i]
                    curr_bool = pat_seizure_bool & (atd_df.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'] == find_str)
                    pat_seizurebool_AND_desiredTypes[curr_bool] = False

            df_subset = atd_df.loc[pat_seizurebool_AND_desiredTypes, ['Type','Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)', 'Date (MM:DD:YYYY)', 'Onset String (HH:MM:SS)', 'Offset String (HH:MM:SS)']]
            
            pat_seiz_startdate_str = df_subset.loc[:,'Date (MM:DD:YYYY)'].astype(str).values.tolist() 
            pat_seiz_starttime_str = df_subset.loc[:,'Onset String (HH:MM:SS)'].astype(str).values.tolist()
            pat_seiz_stoptime_str = df_subset.loc[:,'Offset String (HH:MM:SS)'].astype(str).values.tolist()
            pat_seiz_types_str = df_subset.loc[:,'Seizure Type (FAS; FIAS; FBTC; Non-electrographic; Subclinical; Unknown)'].astype(str).values.tolist()

            # Skip any lines that have nan/none or unknown time entries
            delete_list_A = [i for i, val in enumerate(pat_seiz_starttime_str) if (val=='nan' or val=='Unknown' or val=='None')]
            delete_list_B = [i for i, val in enumerate(pat_seiz_stoptime_str) if (val=='nan' or val=='Unknown' or val=='None')]
            delete_list = list(set(delete_list_A + delete_list_B))
            delete_list.sort()
            if len(delete_list) > 0:
                print(f"WARNING: deleting {len(delete_list)} seizure(s) out of {len(pat_seiz_startdate_str)} due to 'nan'/'none'/'Unknown' in master time sheet")
                print(f"Delete list is: {delete_list}")
                [pat_seiz_startdate_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_starttime_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_stoptime_str.pop(del_idx) for del_idx in reversed(delete_list)]
                [pat_seiz_types_str.pop(del_idx) for del_idx in reversed(delete_list)]

            # Initialize datetimes
            pat_seiz_start_datetimes = [0]*len(pat_seiz_starttime_str)
            pat_seiz_stop_datetimes = [0]*len(pat_seiz_stoptime_str)

            for i in range(0,len(pat_seiz_startdate_str)):
                sD_splits = pat_seiz_startdate_str[i].split(':')
                sT_splits = pat_seiz_starttime_str[i].split(':')
                start_time = datetime.time(
                                    int(sT_splits[0]),
                                    int(sT_splits[1]),
                                    int(sT_splits[2]))
                pat_seiz_start_datetimes[i] = datetime.datetime(int(sD_splits[2]), # Year
                                                    int(sD_splits[0]), # Month
                                                    int(sD_splits[1]), # Day
                                                    int(sT_splits[0]), # Hour
                                                    int(sT_splits[1]), # Minute
                                                    int(sT_splits[2])) # Second
                
                sTstop_splits = pat_seiz_stoptime_str[i].split(':')
                stop_time = datetime.time(
                                    int(sTstop_splits[0]),
                                    int(sTstop_splits[1]),
                                    int(sTstop_splits[2]))

                if stop_time > start_time: # if within same day (i.e. the TIME advances, no date included), assign same date to datetime, otherwise assign next day
                    pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i], stop_time)
                else: 
                    pat_seiz_stop_datetimes[i] = datetime.datetime.combine(pat_seiz_start_datetimes[i] + datetime.timedelta(days=1), stop_time)

            return pat_seiz_start_datetimes, pat_seiz_stop_datetimes, pat_seiz_types_str

        except Exception as e:
            failure_count += 1
            print(f"[Attempt {i+1}] Plotting failed with error: {e}")
            print(f"Total failures so far: {failure_count}")

def get_pat_sleep_datetimes(
    pat_id,
    sleep_file,
    wake_bool = True,
    n1_bool = True,
    n2_bool = True,
    rem_bool = True):

    sleep_df = pd.read_csv(sleep_file, sep='\t', header='infer', on_bad_lines='warn')
    pat_bool = (sleep_df['PatID'] == pat_id) & (sleep_df['Type'] == "Sleep")
    keep_bool = pat_bool

    # Look for each skeep type individually & delete if not desired
    sleep_type_list = ['W', 'N1', 'N2', 'R'] # Must match csv strings
    sleep_type_bool_list = [wake_bool, n1_bool, n2_bool, rem_bool]
    for i in range(0,len(sleep_type_list)):
        if sleep_type_bool_list[i]==False:
            find_str = sleep_type_list[i]
            curr_bool = pat_bool & (sleep_df.loc[:,'SleepCat'] == find_str)
            keep_bool[curr_bool] = False

    df_subset = sleep_df.loc[keep_bool, ['Type','SleepCat', 'OnsetDatetime', 'OffsetDatetime']]
    
    pat_sleep_start_datetime_str = df_subset.loc[:,'OnsetDatetime'].astype(str).values.tolist()
    pat_sleep_stop_datetime_str = df_subset.loc[:,'OffsetDatetime'].astype(str).values.tolist()
    pat_sleep_types_str = df_subset.loc[:,'SleepCat'].astype(str).values.tolist()

    # Convert to datetime objects
    pat_sleep_start_datetimes = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in pat_sleep_start_datetime_str]
    pat_sleep_stop_datetimes = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in pat_sleep_stop_datetime_str]

    return pat_sleep_start_datetimes, pat_sleep_stop_datetimes, pat_sleep_types_str    

def preictal_sleep_label(atd_file, sleep_file, skip_sleep, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input):
    
    '''
    PRE_ICTAL:
    
    Ictal data point is currently labeled as weight of 0

    SLEEP:

    -1 unlabeled
    0 Wake
    1 N1
    2 N2
    3 N3
    4 REM

    '''
    data_window_preictal_score = np.zeros_like(pat_ids_input, dtype=float)
    data_window_ictal_score = np.zeros_like(pat_ids_input, dtype=float)

    data_window_sleep_score = np.ones_like(pat_ids_input, dtype=float) * -1
    sleep_stage_strings = ['W', 'N1', 'N2', 'N3', 'R']
    sleep_stage_to_numeric = {stage: index for index, stage in enumerate(sleep_stage_strings)}
    sleep_epochs_found = 0

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_input))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_input]

    # Initialzie all patient pre-ictal and sleep labels
    num_ids = len(unique_ids)
    pat_seiz_start_datetimes = [-1] * num_ids
    pat_seiz_stop_datetimes = [-1] * num_ids
    pat_seiz_types_str = [-1] * num_ids
    sleep_start_datetimes = [-1] * num_ids
    sleep_stop_datetimes = [-1] * num_ids
    sleep_types = [-1] * num_ids

    # Iterate through unique pats and get all seizure info and sleep info
    for i in range(len(unique_ids)):
        id_curr = unique_ids[i]
        idx_curr = id_to_index[id_curr]
        pat_seiz_start_datetimes[i], pat_seiz_stop_datetimes[i], pat_seiz_types_str[i] = get_pat_seiz_datetimes(id_curr, atd_file) 
        sleep_start_datetimes[i], sleep_stop_datetimes[i], sleep_types[i] = get_pat_sleep_datetimes(id_curr, sleep_file)

    # Iterate through every data window and get pre-ictal and sleep labels
    for i in range(len(pat_idxs)):

        print(f"Getting Pre-Ictal and Sleep Weighting for Data Windows: {i}/{len(pat_idxs)}     ", end='\r')

        data_window_start = start_datetimes_input[i]
        data_window_stop = stop_datetimes_input[i]
        
        # PRE-ICTAL LABELING

        seiz_starts_curr, seiz_stops_curr, seiz_types_curr = pat_seiz_start_datetimes[pat_idxs[i]], pat_seiz_stop_datetimes[pat_idxs[i]], pat_seiz_types_str[pat_idxs[i]]

        for j in range(len(seiz_starts_curr)):
            seiz_start = seiz_starts_curr[j]
            seiz_stop = seiz_stops_curr[j]
            buffered_preictal_start = seiz_start - datetime.timedelta(seconds=plot_preictal_color_sec)

            # Compute time difference from seizure start (0 at start of preictal buffer, increasing as closer to seizure)
            # NOTE: IMPORTANT: Assign value of 0 for in seizure

            # Case where end of data window is in pre-ictal buffer (ignore start of preictal window)
            if data_window_stop < seiz_start and data_window_stop > buffered_preictal_start:
                dist_to_seizure = (seiz_start - data_window_stop).total_seconds()
                new_score = 1.0 - (dist_to_seizure / plot_preictal_color_sec)
                data_window_preictal_score[i] = max(data_window_preictal_score[i], new_score)  
                # Keep max score, and do NOT break because could find higher value

            # Case where end of the window is in the seizure
            elif data_window_stop > seiz_start and data_window_stop < seiz_stop:
                data_window_preictal_score[i] = 0 
                data_window_ictal_score[i] = 1
                break # Want to exclude seizures

            # Case where start of the window overlaps the preictal/ictal buffer, but end is past seizure end
            elif data_window_start > buffered_preictal_start and data_window_start < seiz_stop:
                data_window_preictal_score[i] = 0
                data_window_ictal_score[i] = 1
                break # Want to exclude seizures


        # SLEEP STAGE LABELING

        if not skip_sleep:

            sleep_starts_curr, sleep_stops_curr, sleep_types_curr = sleep_start_datetimes[pat_idxs[i]], sleep_stop_datetimes[pat_idxs[i]], sleep_types[pat_idxs[i]]
            sleep_types_NUM_curr = [sleep_stage_to_numeric[stage] for stage in sleep_types_curr]

            # Find which sleep stage, if none found, leave as initialized value of -1
            for j in range(len(sleep_starts_curr)):
                sleep_start = sleep_starts_curr[j]
                sleep_stop = sleep_stops_curr[j]

                if (data_window_start > sleep_start) & (data_window_stop < sleep_stop):
                    data_window_sleep_score[i] = sleep_types_NUM_curr[j]
                    sleep_epochs_found += 1
                    break

    # Ensure values remain between 0 and 1
    return np.clip(data_window_preictal_score, 0, 1), np.clip(data_window_ictal_score, 0, 1), data_window_sleep_score

def one_hot_encode_with_negatives(data, min_val, max_val):
    """
    One-hot encodes a list of integers, handling negative values.

    Args:
        data: A list of integers.
        min_val: The minimum integer value in the one-hot range.
        max_val: The maximum integer value in the one-hot range.

    Returns:
        A NumPy array representing the one-hot encoded data.
    """
    # Calculate the number of possible values (categories)
    num_classes = max_val - min_val + 1

    # Initialize an empty array to store the one-hot encoded data
    one_hot_data = np.zeros((len(data), num_classes), dtype=int)

    # Create a mapping from the original values to the one-hot indices
    # This is crucial for handling negative values correctly
    value_to_index = {val: val - min_val for val in range(min_val, max_val + 1)}

    # Iterate through the input data and set the corresponding one-hot bit
    for i, val in enumerate(data):
        # Use the mapping to get the correct index
        one_hot_index = value_to_index[val]
        one_hot_data[i, one_hot_index] = 1

    return one_hot_data

def pacmap_subfunction(  
    atd_file,
    pat_ids_list,
    subsample_observation_factor,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    pacmap_LR,
    pacmap_NumIters,
    pacmap_NN,
    pacmap_MN_ratio,
    pacmap_FP_ratio,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    apply_pca=True,
    interictal_contour=False,
    verbose=True,
    xy_lims = [],
    xy_lims_RAW_DIMS = [],
    xy_lims_PCA = [],
    premade_PaCMAP = [],
    premade_PaCMAP_MedDim = [],
    premade_PCA = [],
    premade_HDBSCAN = [],
    exclude_self_pats=False,
    **kwargs):

    '''
    Goal of function:
    Make 2D PaCMAP, make HigherDim PaCMAP, HDBSCAN cluster on HigherDim, visualize clusters on 2D

    '''

    # Metadata
    latent_dim = latent_data_windowed[0].shape[1]
    num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(len(latent_data_windowed)):
        if np.sum(np.isnan(latent_data_windowed[i])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Flatten data into [miniepoch, dim] to feed into PaCMAP, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_PaCMAP_input = np.concatenate(latent_data_windowed, axis=0)

    # Subsample at observation level
    M, N = latent_PaCMAP_input.shape
    num_samples = M // subsample_observation_factor # Compute number of samples to keep
    random_indices = np.random.choice(M, size=num_samples, replace=False) # Randomly choose indices without replacement
    random_indices = np.sort(random_indices) # Optional: sort indices to maintain temporal order
    latent_PaCMAP_input_subsampled = latent_PaCMAP_input[random_indices] # Subsample
    print(f"SHAPE OF PACMAP INPUT: {latent_PaCMAP_input.shape}")
    print(f"SHAPE OF PACMAP INPUT SUBSAMPLED: {latent_PaCMAP_input_subsampled.shape}")

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_list))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_list]
    pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]

    ### PaCMAP 2-Dim ###

    # Make new PaCMAP
    if premade_PaCMAP == []:
        print("Making new 2-dim PaCMAP to use for visualization")
        # initializing the pacmap instance
        # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
        reducer = pacmap.PaCMAP(
            exclude_self_pat=exclude_self_pats,
            pat_idxs=pat_idxs_expanded,
            distance='angular',
            lr=pacmap_LR,
            num_iters=pacmap_NumIters, # will default ~27 if left as None
            n_components=2, 
            n_neighbors=pacmap_NN, # default None, 
            MN_ratio=pacmap_MN_ratio, # default 0.5, 
            FP_ratio=pacmap_FP_ratio, # default 2.0,
            save_tree=True, # Save tree to enable 'transform" method
            apply_pca=apply_pca, 
            verbose=verbose)  

        # fit the data (The index of transformed data corresponds to the index of the original data)
        reducer.fit(latent_PaCMAP_input_subsampled, init='pca')
        print("pacmap fit to subsampled data, now running entire data through fit model")
        latent_postPaCMAP_perfile = reducer.transform(latent_PaCMAP_input)
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    # Use premade PaCMAP
    else: 
        print("Using existing 2-dim PaCMAP for visualization")
        reducer = premade_PaCMAP  # Project data through reducer (i.e. PaCMAP) and split back into file
        latent_postPaCMAP_perfile = reducer.transform(latent_PaCMAP_input)
        latent_postPaCMAP_perfile = np.stack(np.split(latent_postPaCMAP_perfile, len(latent_data_windowed), axis=0),axis=0)

    ### tmp PaCMAP SAVE ####
    pacmap.save(reducer, 'tmp_pacmap')
    print("Saved tmp pacmap")


    ### HDBSCAN ###
    # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
    if premade_HDBSCAN == []:
        # Now do the clustering with HDBSCAN
        print("Building new HDBSCAN model")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_min_cluster_size,
            min_samples=HDBSCAN_min_samples,
            max_cluster_size=0,
            metric='euclidean',  # cosine, manhattan
            # memory=Memory(None, verbose=1)
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
            )
        
        hdb.fit(latent_postPaCMAP_perfile.reshape(latent_postPaCMAP_perfile.shape[0]*latent_postPaCMAP_perfile.shape[1], latent_postPaCMAP_perfile.shape[2]))  # []

         #TODO Look into soft clustering
        # soft_cluster_vecs = np.array(hdbscan.all_points_membership_vectors(hdb))
        # soft_clusters = np.array([np.argmax(x) for x in soft_cluster_vecs], dtype=int)
        # hdb_color_palette = sns.color_palette('Paired', int(np.max(soft_clusters) + 3))

        hdb_labels_flat = hdb.labels_
        # hdb_labels_flat = soft_clusters
        hdb_probabilities_flat = hdb.probabilities_
        # hdb_probabilities_flat = np.array([np.max(x) for x in soft_cluster_vecs])
                
    # If HDBSCAN is already made/provided, then predict cluster with built in HDBSCAN method
    else:
        print("Using pre-built HDBSCAN model")
        hdb = premade_HDBSCAN
        

    #TODO Destaurate according to probability of being in cluster

    # Per patient, Run data through model & Reshape the labels and probabilities for plotting
    hdb_labels_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    hdb_probabilities_flat_perfile = [-1] * latent_postPaCMAP_perfile.shape[0]
    for i in range(len(latent_postPaCMAP_perfile)):
        hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postPaCMAP_perfile[i, :, :])


    ###### START OF PLOTTING #####

    # Get all of the seizure times and types
    seiz_start_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_stop_dt_perfile = [-1] * len(latent_postPaCMAP_perfile)
    seiz_types_perfile = [-1] * len(latent_postPaCMAP_perfile)
    for i in range(len(latent_postPaCMAP_perfile)):
        seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

    # Intialize master figure 
    fig = pl.figure(figsize=(40, 15))
    gs = gridspec.GridSpec(1, 5, figure=fig)


    # **** PACMAP PLOTTING ****

    failure_count = 0
    for i in range(5):
        try:
            print(f"PaCMAP Plotting")
            ax20 = fig.add_subplot(gs[0, 0]) 
            ax21 = fig.add_subplot(gs[0, 1]) 
            ax22 = fig.add_subplot(gs[0, 2]) 
            ax23 = fig.add_subplot(gs[0, 3]) 
            ax24 = fig.add_subplot(gs[0, 4]) 
            ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
                ax=ax20, 
                interCont_ax=ax21,
                seiztype_ax=ax22,
                time_ax=ax23,
                cluster_ax=ax24,
                latent_data=latent_postPaCMAP_perfile.swapaxes(1,2), # [epoch, 2, timesample]
                modified_samp_freq=modified_FS,  # accounts for windowing/stride
                start_datetimes=start_datetimes_epoch, 
                stop_datetimes=stop_datetimes_epoch, 
                win_sec=win_sec,
                stride_sec=stride_sec, 
                seiz_start_dt=seiz_start_dt_perfile, 
                seiz_stop_dt=seiz_stop_dt_perfile, 
                seiz_types=seiz_types_perfile,
                preictal_dur=plot_preictal_color_sec,
                postictal_dur=plot_postictal_color_sec,
                plot_ictal=True,
                hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
                hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
                hdb=hdb,
                xy_lims=xy_lims,
                **kwargs)        

            ax20.title.set_text('PaCMAP Latent Space: ' + 
                'Window mean, dur/str=' + str(win_sec) + 
                '/' + str(stride_sec) +' seconds,' + 
                f'\nLR: {str(pacmap_LR)}, ' +
                f'NumIters: {str(pacmap_NumIters)}, ' +
                f'NN: {pacmap_NN}, MN_ratio: {str(pacmap_MN_ratio)}, FP_ratio: {str(pacmap_FP_ratio)}'
                )
            
            if interictal_contour:
                ax21.title.set_text('Interictal Contour (no peri-ictal data)')

            # **** Save entire figure *****
            if not os.path.exists(savedir): os.makedirs(savedir)
            savename_jpg = savedir + f"/pacmap_latent_smoothsec" + str(win_sec) + "Stride" + str(stride_sec) + "_LR" + str(pacmap_LR) + "_NumIters" + str(pacmap_NumIters) + f"PCA{apply_pca}_NN{pacmap_NN}_MNratio{pacmap_MN_ratio}_FPratio{pacmap_FP_ratio}.jpg"
            pl.savefig(savename_jpg, dpi=600)

            # TODO Upload to WandB

            pl.close(fig)

            print(f"Plotting succeeded on attempt {i+1}")
            break  # Exit the loop if plotting succeeds

        except Exception as e:
            failure_count += 1
            print(f"[Attempt {i+1}] Plotting failed with error: {e}")
            print(f"Total failures so far: {failure_count}")

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
    return ax21, reducer, hdb, xy_lims # save_tuple

def save_pacmap_objects(pacmap_dir, axis, reducer, hdb, xy_lims):

    if not os.path.exists(pacmap_dir): os.makedirs(pacmap_dir) 

    # Save the PaCMAP model for use in inference
    PaCMAP_common_prefix = pacmap_dir + "/PaCMAP"
    pacmap.save(reducer, PaCMAP_common_prefix)
    print("Saved PaCMAP 2-dim model")

    hdbscan_path = pacmap_dir + "/hdbscan.pkl"
    output_obj4 = open(hdbscan_path, 'wb')
    pickle.dump(hdb, output_obj4)
    output_obj4.close()
    print("Saved HDBSCAN model")

    xylim_path = pacmap_dir + "/xy_lims.pkl"
    output_obj7 = open(xylim_path, 'wb')
    pickle.dump(xy_lims, output_obj7)
    output_obj7.close()
    print("Saved xy_lims for PaCMAP")

    axis_path = pacmap_dir + "/plotaxis.pkl"
    output_obj10 = open(axis_path, 'wb')
    pickle.dump(axis, output_obj10)
    output_obj10.close()
    print("Saved plot axis")

def load_pacmap_objects(pretrained_pacmap_dir, pacmap_basename):
    PaCMAP_common_prefix = pretrained_pacmap_dir + f"/{pacmap_basename}"
    reducer = pacmap.load(PaCMAP_common_prefix)

    hdb_file = pretrained_pacmap_dir + f"/hdbscan.pkl"
    with open(hdb_file, "rb") as f: hdb = pickle.load(f)

    xylims_file = pretrained_pacmap_dir + f"/xy_lims.pkl"
    with open(xylims_file, "rb") as f: xy_lims = pickle.load(f)

    plotaxis_file = pretrained_pacmap_dir + f"/plotaxis.pkl"
    with open(plotaxis_file, "rb") as f: plotaxis = pickle.load(f)

    return reducer, hdb, xy_lims, plotaxis

def phate_subfunction(  
    atd_file,
    pat_ids_list,
    subsample_observation_factor,
    latent_data_windowed, 
    start_datetimes_epoch,  
    stop_datetimes_epoch,
    epoch, 
    FS, 
    win_sec, 
    stride_sec, 
    savedir,
    HDBSCAN_min_cluster_size,
    HDBSCAN_min_samples,
    plot_preictal_color_sec,
    plot_postictal_color_sec,
    interictal_contour=False,
    verbose=True,
    knn=50, # Default 5
    decay=10,  # Default 40
    phate_metric='cosine', # 'cosine',
    phate_solver= 'smacof', # 'smacof',
    t='auto',
    gamma=1,
    apply_pca_phate=True,
    pca_comp_phate = 100,
    xy_lims = [],
    custom_nn_bool = False,
    phate_annoy_tree_size = 20,
    knn_indices = [],
    knn_distances = [],
    premade_PHATE = [],
    premade_HDBSCAN = [],
    plot_pat_ids = [],
    store_nn = True,
    **kwargs):

    '''
    Goal of function:
    Run PHATE and plot

    '''

    # Metadata
    latent_dim = latent_data_windowed[0].shape[1]
    num_timepoints_in_windowed_file = latent_data_windowed[0].shape[0]
    modified_FS = 1 / stride_sec

    # Check for NaNs in files
    delete_file_idxs = []
    for i in range(len(latent_data_windowed)):
        if np.sum(np.isnan(latent_data_windowed[i])) > 0:
            delete_file_idxs = delete_file_idxs + [i]
            print(f"WARNING: Deleted file {start_datetimes_epoch[i]} that had NaNs")

    # Delete entries/files in lists where there is NaN in latent space for that file
    latent_data_windowed = [item for i, item in enumerate(latent_data_windowed) if i not in delete_file_idxs]
    start_datetimes_epoch = [item for i, item in enumerate(start_datetimes_epoch) if i not in delete_file_idxs]  
    stop_datetimes_epoch = [item for i, item in enumerate(stop_datetimes_epoch) if i not in delete_file_idxs]
    pat_ids_list = [item for i, item in enumerate(pat_ids_list) if i not in delete_file_idxs]

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_list))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_list]
    pat_idxs_expanded = [item for item in pat_idxs for _ in range(latent_data_windowed[0].shape[0])]


    ### PHATE ###

    # Flatten data into [miniepoch, dim] to feed into PHATE, original data is [file, seq_miniepoch_in_file, latent_dim]
    latent_PHATE_input = np.concatenate(latent_data_windowed, axis=0)

    # Subsample at observation level
    M, N = latent_PHATE_input.shape
    num_samples = M // subsample_observation_factor # Compute number of samples to keep
    random_indices = np.random.choice(M, size=num_samples, replace=False) # Randomly choose indices without replacement
    random_indices = np.sort(random_indices) # Optional: sort indices to maintain temporal order
    latent_PHATE_input_subsampled = latent_PHATE_input[random_indices] # Subsample
    print(f"SHAPE OF PHATE INPUT: {latent_PHATE_input.shape}")
    print(f"SHAPE OF PHATE INPUT SUBSAMPLED: {latent_PHATE_input_subsampled.shape}")

    # No PHATE object passed in, make new one
    if premade_PHATE == []:
        # Default PHATE NN search
        print("DEFAULT PHATE NN Search")
        # Build and fit default PHATE object
        phate_op = phate.PHATE(
            knn=knn,
            knn_max=knn*5,
            decay=decay,
            gamma=gamma,
            t=t,
            mds_dist=phate_metric,
            knn_dist=phate_metric,
            mds_solver=phate_solver,
            n_jobs= -2)
        phate_op.fit(latent_PHATE_input_subsampled) # Pass subsampled data to just FIT the model
        phate_output = phate_op.transform(latent_PHATE_input) # Run all data through

    # Premade PHATE object has been passed in 
    else:
        phate_op = premade_PHATE
        phate_output = phate_op.transform(latent_PHATE_input)

    # Split reduced output back into files
    latent_postPHATE_perfile = np.stack(np.split(phate_output, len(latent_data_windowed), axis=0),axis=0)

    ### HDBSCAN ###
    # If training, create new cluster model, otherwise "approximate_predict()" if running on val data
    if premade_HDBSCAN == []:
        # Now do the clustering with HDBSCAN
        print("Building new HDBSCAN model")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_min_cluster_size,
            min_samples=HDBSCAN_min_samples,
            max_cluster_size=0,
            metric='euclidean',  # cosine, manhattan
            # memory=Memory(None, verbose=1)
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
            )
        
        hdb.fit(latent_postPHATE_perfile.reshape(latent_postPHATE_perfile.shape[0]*latent_postPHATE_perfile.shape[1], latent_postPHATE_perfile.shape[2]))  # []

         #TODO Look into soft clustering
        # soft_cluster_vecs = np.array(hdbscan.all_points_membership_vectors(hdb))
        # soft_clusters = np.array([np.argmax(x) for x in soft_cluster_vecs], dtype=int)
        # hdb_color_palette = sns.color_palette('Paired', int(np.max(soft_clusters) + 3))

        hdb_labels_flat = hdb.labels_
        # hdb_labels_flat = soft_clusters
        hdb_probabilities_flat = hdb.probabilities_
        # hdb_probabilities_flat = np.array([np.max(x) for x in soft_cluster_vecs])
                
    # If HDBSCAN is already made/provided, then predict cluster with built in HDBSCAN method
    else:
        print("Using pre-built HDBSCAN model")
        hdb = premade_HDBSCAN
        
    #TODO Destaurate according to probability of being in cluster

    # Per patient, Run data through model & Reshape the labels and probabilities for plotting
    hdb_labels_flat_perfile = [-1] * latent_postPHATE_perfile.shape[0]
    hdb_probabilities_flat_perfile = [-1] * latent_postPHATE_perfile.shape[0]
    for i in range(len(latent_postPHATE_perfile)):
        hdb_labels_flat_perfile[i], hdb_probabilities_flat_perfile[i] = hdbscan.prediction.approximate_predict(hdb, latent_postPHATE_perfile[i, :, :])


    ###### START OF PLOTTING #####

    # Get all of the seizure times and types
    seiz_start_dt_perfile = [-1] * len(pat_ids_list)
    seiz_stop_dt_perfile = [-1] * len(pat_ids_list)
    seiz_types_perfile = [-1] * len(pat_ids_list)
    for i in range(len(pat_ids_list)):
        seiz_start_dt_perfile[i], seiz_stop_dt_perfile[i], seiz_types_perfile[i] = get_pat_seiz_datetimes(pat_ids_list[i], atd_file=atd_file)

    # Intialize master figure 
    fig_height = 15 + 5 * len(plot_pat_ids)
    fig = pl.figure(figsize=(40, fig_height))
    gs = gridspec.GridSpec(1 + len(plot_pat_ids), 5, figure=fig)

    # **** PHATE PLOTTING ****

    print(f"PHATE Plotting")
    ax00 = fig.add_subplot(gs[0, 0]) 
    ax01 = fig.add_subplot(gs[0, 1]) 
    ax02 = fig.add_subplot(gs[0, 2]) 
    ax03 = fig.add_subplot(gs[0, 3]) 
    ax04 = fig.add_subplot(gs[0, 4]) 
    ax00, ax01, ax02, ax03, ax04, xy_lims = plot_latent(
        ax=ax00, 
        interCont_ax=ax01,
        seiztype_ax=ax02,
        time_ax=ax03,
        cluster_ax=ax04,
        latent_data=latent_postPHATE_perfile.swapaxes(1,2), # [epoch, 2, timesample]
        modified_samp_freq=modified_FS,  # accounts for windowing/stride
        start_datetimes=start_datetimes_epoch, 
        stop_datetimes=stop_datetimes_epoch, 
        win_sec=win_sec,
        stride_sec=stride_sec, 
        seiz_start_dt=seiz_start_dt_perfile, 
        seiz_stop_dt=seiz_stop_dt_perfile, 
        seiz_types=seiz_types_perfile,
        preictal_dur=plot_preictal_color_sec,
        postictal_dur=plot_postictal_color_sec,
        plot_ictal=True,
        hdb_labels=np.expand_dims(np.stack(hdb_labels_flat_perfile, axis=0),axis=1),
        hdb_probabilities=np.expand_dims(np.stack(hdb_probabilities_flat_perfile, axis=0),axis=1),
        hdb=hdb,
        xy_lims=xy_lims,
        **kwargs)        

    ax00.title.set_text('PHATE Latent Space: ' + 
        'Window mean, dur/str=' + str(win_sec) + 
        '/' + str(stride_sec) +' seconds' )
    
    if interictal_contour:
        ax01.title.set_text('Interictal Contour (no peri-ictal data)')


    #### Plot the individual patient IDs defined in 'plot_pat_ids'
    for i in range(len(plot_pat_ids)):
        print(f"Patient Specific PHATE Plotting")
        pat_id_curr = plot_pat_ids[i]
        
        # Subindex the patient's data
        found_file_ids = [index for index, value in enumerate(pat_ids_list) if value == pat_id_curr]
        pat_data = latent_postPHATE_perfile[found_file_ids] 
        pat_start_datetimes_epoch = [start_datetimes_epoch[x] for x in found_file_ids]
        pat_stop_datetimes_epoch = [stop_datetimes_epoch[x] for x in found_file_ids]
        pat_seiz_start_dt_perfile = [seiz_start_dt_perfile[x] for x in found_file_ids]
        pat_seiz_stop_dt_perfile = [seiz_stop_dt_perfile[x] for x in found_file_ids]
        pat_seiz_types_perfile = [seiz_types_perfile[x] for x in found_file_ids]
        pat_hdb_labels_flat_perfile = [hdb_labels_flat_perfile[x] for x in found_file_ids]
        pat_hdb_probabilities_flat_perfile = [hdb_probabilities_flat_perfile[x] for x in found_file_ids]

        # Make the patient's subplots
        axi0 = fig.add_subplot(gs[1 + i, 0]) 
        axi1 = fig.add_subplot(gs[1 + i, 1]) 
        axi2 = fig.add_subplot(gs[1 + i, 2]) 
        axi3 = fig.add_subplot(gs[1 + i, 3]) 
        axi4 = fig.add_subplot(gs[1 + i, 4]) 

        ax20, ax21, ax22, ax23, ax24, xy_lims = plot_latent(
            ax=axi0, 
            interCont_ax=axi1,
            seiztype_ax=axi2,
            time_ax=axi3,
            cluster_ax=axi4,
            latent_data=pat_data.swapaxes(1,2), # [epoch, 2, timesample]
            modified_samp_freq=modified_FS,  # accounts for windowing/stride
            start_datetimes=pat_start_datetimes_epoch, 
            stop_datetimes=pat_stop_datetimes_epoch, 
            win_sec=win_sec,
            stride_sec=stride_sec, 
            seiz_start_dt=pat_seiz_start_dt_perfile, 
            seiz_stop_dt=pat_seiz_stop_dt_perfile, 
            seiz_types=pat_seiz_types_perfile,
            preictal_dur=plot_preictal_color_sec,
            postictal_dur=plot_postictal_color_sec,
            plot_ictal=True,
            hdb_labels=np.expand_dims(np.stack(pat_hdb_labels_flat_perfile, axis=0),axis=1),
            hdb_probabilities=np.expand_dims(np.stack(pat_hdb_probabilities_flat_perfile, axis=0),axis=1),
            hdb=hdb,
            xy_lims=xy_lims, # passed from above
            **kwargs)     


        axi0.title.set_text(f"Only {pat_id_curr}")


        failure_count = 0
    
    for i in range(5):
        try:
            # **** Save entire figure *****
            if not os.path.exists(savedir): os.makedirs(savedir)
            # if not os.path.exists(savedir + '/PDFs'): os.makedirs(savedir + '/PDFs')
            savename_jpg = savedir + f"/PHATE_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{phate_metric}_knn{knn}_decay{decay}.jpg"
            # savename_pdf = savedir + f"/PDFs/PHATE_latent_smoothsec{win_sec}Stride{stride_sec}_epoch{epoch}_{phate_metric}_knn{knn}_decay{decay}.pdf"
            pl.savefig(savename_jpg, dpi=600)
            # pl.savefig(savename_pdf, dpi=600)

            # TODO Upload to WandB

            pl.close(fig)

            print(f"Plotting succeeded on attempt {i+1}")
            break  # Exit the loop if plotting succeeds

        except Exception as e:
            failure_count += 1
            print(f"[Attempt {i+1}] Plotting failed with error: {e}")
            print(f"Total failures so far: {failure_count}")

    # Bundle the save metrics together
    # save_tuple = (latent_data_windowed.swapaxes(1,2), latent_PCA_allFiles, latent_topPaCMAP_allFiles, latent_topPaCMAP_MedDim_allFiles, hdb_labels_allFiles, hdb_probabilities_allFiles)
    return ax01, phate_op, hdb, xy_lims # save_tuple
    

def save_phate_objects(phate_dir, axis, reducer, hdb, xy_lims):

    if not os.path.exists(phate_dir): os.makedirs(phate_dir) 

    # Save the PaCMAP model for use in inference
    phate_path = phate_dir + "/phate.pkl"
    output_obj4 = open(phate_path, 'wb')
    pickle.dump(reducer, output_obj4)
    output_obj4.close()
    print("Saved PHATE model")

    hdbscan_path = phate_dir + "/hdbscan.pkl"
    output_obj4 = open(hdbscan_path, 'wb')
    pickle.dump(hdb, output_obj4)
    output_obj4.close()
    print("Saved HDBSCAN model")

    xylim_path = phate_dir + "/xy_lims.pkl"
    output_obj7 = open(xylim_path, 'wb')
    pickle.dump(xy_lims, output_obj7)
    output_obj7.close()
    print("Saved xy_lims for PHATE")

    axis_path = phate_dir + "/plotaxis.pkl"
    output_obj10 = open(axis_path, 'wb')
    pickle.dump(axis, output_obj10)
    output_obj10.close()
    print("Saved PHATE plot axis")

def load_phate_objects(pretrained_phate_dir):
    phate_file = pretrained_phate_dir + f"/phate.pkl"
    with open(phate_file, "rb") as f: reducer = pickle.load(f)

    hdb_file = pretrained_phate_dir + f"/hdbscan.pkl"
    with open(hdb_file, "rb") as f: hdb = pickle.load(f)

    xylims_file = pretrained_phate_dir + f"/xy_lims.pkl"
    with open(xylims_file, "rb") as f: xy_lims = pickle.load(f)

    plotaxis_file = pretrained_phate_dir + f"/plotaxis.pkl"
    with open(plotaxis_file, "rb") as f: plotaxis = pickle.load(f)

    return reducer, hdb, xy_lims, plotaxis