import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import math

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

def select_cim_sigma(data, subsample=1000, scale=1.0, 
                     scales_to_test=None, bins=100, 
                     metric='entropy', plot=True, savename='cim_overlayed_histograms.png', verbose=True):
    """
    Unified function to:
      - Estimate a base CIM σ using median pairwise distances
      - Select optimal σ based on entropy/variance of correntropy histograms
      - Optionally plot overlayed histograms

    Returns:
      best_sigma, best_scale, best_score, all_scores
    """

    # Step 1: Subsample
    if data.shape[0] > subsample:
        idx = np.random.choice(data.shape[0], size=subsample, replace=False)
        sample = data[idx]
    else:
        sample = data

    # Step 2: Estimate base sigma from pairwise distances
    dists = pairwise_distances(sample, metric='euclidean')
    i_upper = np.triu_indices_from(dists, k=1)
    dist_values = dists[i_upper]
    base_sigma = scale * np.median(dist_values)

    # Step 3: Define scales to test
    if scales_to_test is None:
        scales_to_test = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    results = []
    for s in scales_to_test:
        sigma = base_sigma * s
        correntropy_values = np.exp(- (dist_values ** 2) / (2 * sigma ** 2))
        hist, _ = np.histogram(correntropy_values, bins=bins, range=(0, 1), density=True)
        if metric == 'entropy':
            score = entropy(hist + 1e-8)
        elif metric == 'variance':
            score = np.var(hist)
        else:
            raise ValueError("Metric must be 'entropy' or 'variance'")
        results.append((sigma, s, score))

    # Select best
    results.sort(key=lambda x: -x[2])
    best_sigma, best_scale, best_score = results[0]

    # Optional plot
    if plot:
        # Automatically generate enough distinct colors
        cmap = cm.get_cmap('tab20', len(scales_to_test))
        color_list = [cmap(i) for i in range(len(scales_to_test))]

        plt.figure(figsize=(8, 5))
        for i, s in enumerate(scales_to_test):
            sigma = base_sigma * s
            correntropy_values = np.exp(- (dist_values ** 2) / (2 * sigma ** 2))
            label = f"{int(s * 100)}% σ (σ = {sigma:.4f})"
            plt.hist(correntropy_values, bins=bins, alpha=0.4, 
                     color=color_list[i], label=label, density=True)

        plt.title(f'Correntropy Histograms for Scaled σ Values\nBase σ = {base_sigma:.4f}')
        plt.xlabel('Correntropy Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(savename, dpi=600)

    if verbose:
        # Sort for printing
        results_sorted = sorted(results, key=lambda x: x[0])  # sort by sigma

        print("Tried sigmas (sorted ascending) and corresponding entropy/variance values:")
        for sigma, s, score in results_sorted:
            print(f"Sigma: {sigma:.4f}, Scale: {s:.2f}, {metric.capitalize()}: {score:.4f}")

        print(f"\nBest CIM selected: Sigma = {best_sigma:.4f}, Scale = {best_scale:.2f}, {metric.capitalize()} = {best_score:.4f}")


    return best_sigma, best_scale, best_score, results


class ToroidalSOM_2(nn.Module):
    """
    Optimized Toroidal Self-Organizing Map (SOM) with hexagonal geometry in PyTorch.
    Optimized for GPU performance.
    """

    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, cim_kernel_sigma, 
                 sigma, sigma_epoch_decay, sigma_min, data_for_init, device, pca, **kwargs):
        super(ToroidalSOM_2, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lr = lr
        self.lr_epoch_decay = lr_epoch_decay
        self.cim_kernel_sigma = cim_kernel_sigma
        self.sigma = sigma
        self.sigma_epoch_decay = sigma_epoch_decay
        self.sigma_min = sigma_min
        self.device = device
        self.pca = pca
        self.data_for_init = data_for_init

        # Move PCA components to GPU if PCA passed in
        if (pca is not None): 
            self.pca_components = torch.tensor(pca.components_, dtype=torch.float32).to(device)  # shape (n_components, input_dim)
        else: 
            self.pca_components = None

        # Create the hexagonal coordinates and weights
        self.hex_coords = self._create_hex_grid()
        if data_for_init != None: self.weights = self._initialize_weights()

    def _create_hex_grid(self):
        """
        Create a hexagonal grid coordinates.
        Returns the hexagonal coordinates as a PyTorch tensor on the specified device.
        """
        rows, cols = self.grid_size
        hex_coords = torch.zeros(rows, cols, 2, device=self.device)
        for i in range(rows):
            for j in range(cols):
                x = j + (0.5 * (i % 2))
                y = i * 0.866
                hex_coords[i, j, 0] = x
                hex_coords[i, j, 1] = y
        return hex_coords

    def _apply_pca_projection(self, data_gpu):
        """
        Projects GPU-resident data using precomputed PCA components.
        Args:
            data_gpu (torch.Tensor): (N, D) input data on GPU
            self.pca_components_gpu (torch.Tensor): (n_components, D) PCA basis on GPU
        Returns:
            projected: (N, n_components) projection result
        """
        if self.pca_components is None: 
            return data_gpu
        else: 
            return data_gpu @ self.pca_components.T
            

    def _initialize_weights(self):
        """Initialize weights from random data points."""
        print("Initializing weights with random selection of input data...")
        unique_random_integers = np.random.choice(range(self.data_for_init.shape[0]), size=self.grid_size[0]*self.grid_size[1], replace=True)
        rand_data_selection = self.data_for_init[unique_random_integers].reshape(self.grid_size[0],self.grid_size[1],-1)

        # return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)
        return torch.tensor(rand_data_selection, dtype=torch.float32, device=self.device)

    def reset_device(self, device):
        """Move all tensors to the specified device."""
        self.hex_coords = self.hex_coords.to(device)
        self.weights = self.weights.to(device)
        self.device = device

    def forward(self, x):
        """
        Compute CIM distance between input x and SOM weights.
        CIM(x, y) = sqrt(1 - exp(-||x - y||^2 / (2 * sigma^2)))
        """
        x_expanded = x[:, None, None, :]  # (B, 1, 1, D)
        diff = self.weights - x_expanded  # (B, rows, cols, D)
        sq_dist = torch.sum(diff ** 2, dim=3)  # (B, rows, cols)

        kernel_sigma_sq = 2 * (self.cim_kernel_sigma ** 2)
        sim = torch.exp(-sq_dist / kernel_sigma_sq)  # Similarity
        cim = torch.sqrt(1 - sim + 1e-8)  # CIM distance (add epsilon to avoid sqrt(0))

        return cim

    def find_bmu(self, x):
        """Find Best Matching Units (BMUs) for each input vector."""
        distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
        flat_distances = distances.view(x.size(0), -1)
        bmu_flat_indices = torch.argmin(flat_distances, dim=1)
        bmu_rows = bmu_flat_indices // self.grid_size[1]
        bmu_cols = bmu_flat_indices % self.grid_size[1]
        return bmu_rows, bmu_cols

    def _get_neighbors_toroidal(self, bmu_rows, bmu_cols, sigma):
        """Efficiently get toroidal neighbors within a given radius (sigma)."""
        batch_size = bmu_rows.size(0)
        rows, cols = self.grid_size
        max_dist_sq = (sigma * 1.5)**2 # Heuristic for hexagonal grid

        row_indices = torch.arange(rows, device=self.device).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, cols)
        col_indices = torch.arange(cols, device=self.device).float().unsqueeze(0).unsqueeze(0).repeat(batch_size, rows, 1)

        bmu_row_expanded = bmu_rows.float().unsqueeze(-1).unsqueeze(-1).repeat(1, rows, cols)
        bmu_col_expanded = bmu_cols.float().unsqueeze(-1).unsqueeze(-1).repeat(1, rows, cols)

        row_diff = torch.abs(row_indices - bmu_row_expanded)
        col_diff = torch.abs(col_indices - bmu_col_expanded)

        row_diff = torch.min(row_diff, rows - row_diff)
        col_diff = torch.min(col_diff, cols - col_diff)

        # Adjust for hexagonal grid distance (approximation)
        dist_sq = (col_diff + 0.5 * (row_diff % 2))**2 + (row_diff * 0.866)**2
        neighborhood_mask = (dist_sq < max_dist_sq).float()

        return neighborhood_mask, row_diff, col_diff

    def update_weights(self, x, bmu_rows, bmu_cols):
        """Update SOM weights using toroidal distances and vectorized operations."""
        batch_size = bmu_rows.size(0)
        rows, cols = self.grid_size

        neighborhood_mask, row_diff, col_diff = self._get_neighbors_toroidal(bmu_rows, bmu_cols, self.sigma)

        # Gaussian neighborhood function based on toroidal distance
        dist_sq = (col_diff + 0.5 * (row_diff % 2))**2 + (row_diff * 0.866)**2
        neighborhood = torch.exp(-dist_sq / (2 * self.sigma**2)) * neighborhood_mask

        # Expand dimensions for weight update
        x_expanded = x[:, None, None, :]
        neighborhood_expanded = neighborhood.unsqueeze(-1)

        # Expand weights to (B, rows, cols, D) to match batch-wise updates
        weights_expanded = self.weights.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, rows, cols, D)

        # Compute delta per sample
        delta = self.lr * neighborhood_expanded * (x_expanded - weights_expanded)  # (B, rows, cols, D)

        # Aggregate across batch
        weight_update = torch.sum(delta, dim=0) / batch_size  # (rows, cols, D)

        # Update weights
        self.weights += weight_update


    def sample_posterior(self, batch_means, batch_logvars):
        # Add noise equivelant to magnitude of logvar to estimate uncertainty
        std = torch.exp(0.5 * batch_logvars)  # Standard deviation
        eps = torch.randn_like(std)     # Sample from a standard normal distribution
        z = batch_means + eps * std            # Reparameterized sample

        return self._apply_pca_projection(z) # will just return z if no PCA is used

    def train(self, means_in, logvars_in, num_epochs):
        """Train the SOM on the input data."""
        means_tensor = torch.tensor(means_in, dtype=torch.float32, device=self.device)
        logvars_tensor = torch.tensor(logvars_in, dtype=torch.float32, device=self.device)
        num_samples = means_tensor.shape[0]
        indices = torch.arange(num_samples, device=self.device)

        for epoch in range(num_epochs):
            # Shuffle data indices
            shuffled_indices = indices[torch.randperm(num_samples)]

            # Train in batches
            for batch_start in range(0, num_samples, self.batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + self.batch_size]
                batch_means = means_tensor[batch_indices]
                batch_logvars = logvars_tensor[batch_indices]
                batch_sampled = self.sample_posterior(batch_means, batch_logvars)

                # Find BMUs
                bmu_rows, bmu_cols = self.find_bmu(batch_sampled)

                # Update weights
                self.update_weights(batch_sampled, bmu_rows, bmu_cols)

                # Progress tracking
                batch_num = batch_start // self.batch_size
                total_batches = math.ceil(num_samples / self.batch_size)
                print(f"\rToroidal SOM Epoch: {epoch}/{num_epochs-1}, "
                      f"Batch: {batch_num}/{total_batches} [Batch Size: {self.batch_size}], "
                      f"Iter:{batch_start}/{shuffled_indices.shape[0]-1}, "
                      f"Sigma: {self.sigma:.4f}, LR: {self.lr:.6f}", end="")

            # Decay learning rate and sigma
            self.lr *= self.lr_epoch_decay
            self.sigma = max(self.sigma * self.sigma_epoch_decay, self.sigma_min)

    def get_weights(self):
        """Return weights as a NumPy array."""
        return self.weights.cpu().numpy()

    def get_hex_coords(self):
        """Return the 2D hexagonal coordinates as a NumPy array."""
        return self.hex_coords.cpu().numpy()

    def project_data(self, data):
        """Project data onto the SOM grid and return grid positions."""
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        bmu_rows, bmu_cols = self.find_bmu(data)
        return bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()