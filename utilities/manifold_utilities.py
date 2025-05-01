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
# from sklearn.decomposition import PCA
from models.ToroidalSOM import ToroidalSOM
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

# def get_som_rowcol(data, som):
#     """Helper to run a batch of data through the SOM and get (row, col) coordinates."""
#     som_rowcol = np.zeros((data.shape[0], 2), dtype=np.int32)
#     for i in range(0, len(data), som.batch_size):
#         batch = data[i:i + som.batch_size]
#         batch = torch.tensor(batch, dtype=torch.float32, device=som.device)
#         bmu_rows, bmu_cols = som.find_bmu(batch)
#         bmu_rows, bmu_cols = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()
#         som_rowcol[i:i + som.batch_size, 0] = bmu_rows
#         som_rowcol[i:i + som.batch_size, 1] = bmu_cols
#     return som_rowcol

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

def plot_kohonen_prediction(gpu_id, save_dir, som, plot_data_path, context, ground_truth_future, predictions, undo_log, smoothing_factor, epoch, batch_idx, pat_id, overlay_thresh=0.25):
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
            if overlay_preictal[i, j] >= overlay_thresh:
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
    pl.savefig(savename_overlay, dpi=600)
    pl.close(fig_overlay)

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
    som_precomputed_path = None,
    som_object = None,
    som_device = 0,
    sigma_plot=1,
    hits_log_view=True,
    umat_log_view=True,
    overlay_thresh = 0.25,
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

    else:
        # Make new Toroidal SOM and train it
        grid_size = (som_gridsize, som_gridsize)
        print(f"Training brand new Toroidal SOM: gridsize:{som_gridsize}, lr:{som_lr} w/ {som_lr_epoch_decay:.4f} decay per epoch to {som_lr * (som_lr_epoch_decay ** som_epochs):.6f}, sigma:{som_sigma} w/ {som_sigma_epoch_decay:.4f} decay per epoch to {som_sigma_min:.6f}")
        som = ToroidalSOM(grid_size=grid_size, input_dim=latent_means_input.shape[1], batch_size=som_batch_size, lr=som_lr,
                            lr_epoch_decay=som_lr_epoch_decay, sigma=som_sigma, sigma_epoch_decay=som_sigma_epoch_decay,
                            sigma_min=som_sigma_min, device=som_device, init_pca=som_pca_init, data_for_pca=latent_means_input)

        # Train and save SOM
        som.train(latent_means_input, latent_logvars_input, num_epochs=som_epochs)
        savepath = savedir + f"/GPU{som_device}_ToroidalSOM_ObjectDict_smoothsec{win_sec}_Stride{stride_sec}_subsampleFileFactor{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.6f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.pt"
        torch.save({
            'model_state_dict': som.state_dict(),
            'weights': som.weights,
            'grid_size': som_gridsize,
            'input_dim': latent_means_input.shape[1],
            'lr': som_lr,
            'sigma': som_sigma,
            'lr_epoch_decay': som_lr_epoch_decay,
            'sigma_epoch_decay': som_sigma_epoch_decay,
            'sigma_min': som_sigma_min,
            'epoch': som_epochs,
            'batch_size': som_batch_size
        }, savepath)
        print(f"Toroidal SOM model saved at {savepath}")

    # PLOT PREPARATION

    # Get preictal weights for each data point
    preictal_float_input, ictal_float_input = preictal_weight(atd_file, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input)

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

    # SOM Inference on all data in batches
    for i in range(0, len(latent_means_input), som_batch_size):
        batch = latent_means_input[i:i + som_batch_size]
        batch_patients = pat_ids_input[i:i + som_batch_size]
        batch_preictal_labels = preictal_float_input[i:i + som_batch_size]
        batch_ictal_labels = ictal_float_input[i:i + som_batch_size]

        batch = torch.tensor(batch, dtype=torch.float32, device=som_device)
        bmu_rows, bmu_cols = som.find_bmu(batch)
        bmu_rows, bmu_cols = bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()

        # Update hit map
        np.add.at(hit_map, (bmu_rows, bmu_cols), 1)

        # Accumulate preictal scores
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            preictal_sums[bmu_row, bmu_col] += batch_preictal_labels[j]
            ictal_sums[bmu_row, bmu_col] += batch_ictal_labels[j]

        # Track unique patients per node
        for j, (bmu_row, bmu_col) in enumerate(zip(bmu_rows, bmu_cols)):
            if (bmu_row, bmu_col) not in neuron_patient_dict:
                neuron_patient_dict[(bmu_row, bmu_col)] = set()
            neuron_patient_dict[(bmu_row, bmu_col)].add(batch_patients[j])

    # If hits want to be viewed logarithmically
    if hits_log_view:
        epsilon = np.finfo(float).eps
        hit_map = np.log(hit_map + epsilon)

    # Normalize preictal & ictal sums
    if np.max(preictal_sums) > np.min(preictal_sums):
        preictal_sums = (preictal_sums - np.min(preictal_sums)) / (np.max(preictal_sums) - np.min(preictal_sums))

    if np.max(ictal_sums) > np.min(ictal_sums):
        ictal_sums = (ictal_sums - np.min(ictal_sums)) / (np.max(ictal_sums) - np.min(ictal_sums))

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
    preictal_sums_smoothed = gaussian_filter(preictal_sums, sigma=1.0)
    if np.max(preictal_sums_smoothed) > np.min(preictal_sums_smoothed):
        preictal_sums_smoothed = (preictal_sums_smoothed - np.min(preictal_sums_smoothed)) / (np.max(preictal_sums_smoothed) - np.min(preictal_sums_smoothed))

    # Calculate rescaled map (preictal * patient diversity)
    rescale_preictal = preictal_sums * patient_map
    if np.max(rescale_preictal) > 0:
        rescale_preictal = rescale_preictal / np.max(rescale_preictal)

    # Smooth the rescaled PRE-ICTAL map
    rescale_preictal_smoothed = gaussian_filter(rescale_preictal, sigma=1.0)
    if np.max(rescale_preictal_smoothed) > 0:
        rescale_preictal_smoothed = rescale_preictal_smoothed / np.max(rescale_preictal_smoothed)

    # Smooth the ICTAL map
    ictal_smoothed = gaussian_filter(ictal_sums, sigma=1.0)
    if np.max(ictal_smoothed) > 0:
        ictal_smoothed = ictal_smoothed / np.max(ictal_smoothed)


    # PLOTTING (2D Hexagonal Plots)

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
    savename_jpg_2d = savedir + f"/GPU{som_device}_2DPlots_ToroidalSOM_latent_smoothsec{win_sec}_Stride{stride_sec}_subsampleFileFactor{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.6f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}_HEXAGONAL_2D.jpg"
    pl.savefig(savename_jpg_2d, dpi=600)
    pl.savefig(savename_jpg_2d.replace('.jpg', '.svg'))


    # 2D OVERLAY: U-Matrix + Pre-Ictal
    
    # Create new figure for U-Matrix + Pre-Ictal Density Overlay
    fig_overlay, ax_overlay = pl.subplots(figsize=(10, 10))

    # Clip preictal_sums_smoothed at lower threshold 0.5
    # overlay_preictal = np.clip(preictal_sums_smoothed, 0.0, 1.0)
    overlay_preictal = np.clip(rescale_preictal_smoothed, 0.0, 1.0)
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
            if overlay_preictal[i, j] >= overlay_thresh:
                ax_overlay.add_patch(hexagon)

    # Optional: add a colorbar for the overlay
    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    pl.colorbar(sm_overlay, ax=ax_overlay, label="Pre-Ictal Density (Clipped & Smoothed)")

    # Save overlay figure
    savename_overlay = savedir + f"/GPU{som_device}_UMatrix_PreIctalOverlay__ToroidalSOM_latent_smoothsec{win_sec}_Stride{stride_sec}_subsampleFileFactor{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.6f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}_HEXAGONAL_2D.jpg"
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
            if overlay_ictal[i, j] >= overlay_thresh:
                ax_overlay.add_patch(hexagon)

    # Optional: add a colorbar for the overlay
    sm_overlay = pl.cm.ScalarMappable(cmap=cmap_overlay, norm=norm_overlay)
    sm_overlay.set_array([])
    pl.colorbar(sm_overlay, ax=ax_overlay, label="ICTAL Density")

    # Save overlay figure
    savename_overlay = savedir + f"/GPU{som_device}_UMatrix_ICTAL_Overlay__ToroidalSOM_latent_smoothsec{win_sec}_Stride{stride_sec}_subsampleFileFactor{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.6f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}_HEXAGONAL_2D.jpg"
    pl.savefig(savename_overlay, dpi=600)
    pl.savefig(savename_overlay.replace('.jpg', '.svg'))


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
    savename_jpg_3d = savedir + f"/GPU{som_device}_3DPlots_smoothsec{win_sec}_Stride{stride_sec}_subsampFile{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.4f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.jpg"
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
    savename_jpg_toroid = savedir + f"/GPU{som_device}_ToroidalSOM_latent_smoothsec{win_sec}_Stride{stride_sec}_subsampFile{subsample_file_factor}_preictalSec{plot_preictal_color_sec}_gridsize{som_gridsize}_lr{som_lr}with{som_lr_epoch_decay:.4f}decay{lr_min:0.4f}min_sigma{som_sigma}with{som_sigma_epoch_decay:.4f}decay{som_sigma_min}min_numfeatures{latent_means_input.shape[0]}_dims{latent_means_input.shape[1]}_batchsize{som_batch_size}_epochs{som_epochs}.jpg"
    pl.savefig(savename_jpg_toroid, dpi=600)
    pl.savefig(savename_jpg_toroid.replace('.jpg', '.svg'))
            
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

    # # Debugging
    # print(pat_id)

    # Original ATD file from Derek was tab seperated
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

def preictal_weight(atd_file, plot_preictal_color_sec, pat_ids_input, start_datetimes_input, stop_datetimes_input):
    
    '''
    Ictal is given 0 label - Not the focus of the BSE/SOM training, too rare, likely to give erroneous BSE embeddings/SOM mappings

    '''

    data_window_preictal_score = np.zeros_like(pat_ids_input, dtype=float)
    data_window_ictal_score = np.zeros_like(pat_ids_input, dtype=float)

    # Generate numerical IDs for each unique patient, and give each datapoint an ID
    unique_ids = list(set(pat_ids_input))
    id_to_index = {id: idx for idx, id in enumerate(unique_ids)}  # Create mapping dictionary
    pat_idxs = [id_to_index[id] for id in pat_ids_input]

    pat_seiz_start_datetimes, pat_seiz_stop_datetimes, pat_seiz_types_str = [-1] * len(unique_ids), [-1] * len(unique_ids), [-1] * len(unique_ids)
    for i in range(len(unique_ids)):
        id_curr = unique_ids[i]
        idx_curr = id_to_index[id_curr]
        pat_seiz_start_datetimes[i], pat_seiz_stop_datetimes[i], pat_seiz_types_str[i] = get_pat_seiz_datetimes(id_curr, atd_file) 

    for i in range(len(pat_idxs)):
        data_window_start = start_datetimes_input[i]
        data_window_stop = stop_datetimes_input[i]
        
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

    # Ensure values remain between 0 and 1
    return np.clip(data_window_preictal_score, 0, 1), np.clip(data_window_ictal_score, 0, 1)

