import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math, os
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

class ToroidalSOM_CIM(nn.Module):
    """
    Optimized Toroidal Self-Organizing Map (SOM) with hexagonal geometry in PyTorch.
    Optimized for GPU performance.
    """

    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, cim_kernel_sigma, sigma,
                 sigma_epoch_decay, sigma_min, device, entropy_penalty, winner_penalty, bmu_count_decay, temperature, init_pca=False, data_for_init=None, **kwargs):
        super(ToroidalSOM_CIM, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lr = lr
        self.lr_epoch_decay = lr_epoch_decay
        self.cim_kernel_sigma = cim_kernel_sigma
        self.sigma = sigma
        self.sigma_epoch_decay = sigma_epoch_decay
        self.sigma_min = sigma_min
        self.entropy_penalty = entropy_penalty
        self.winner_penalty = winner_penalty
        self.bmu_count_decay = bmu_count_decay
        self.temperature = temperature
        self.device = device
        self.init_pca = init_pca
        self.data_for_init = data_for_init

        # To track long-term node usage
        self.bmu_counts = torch.zeros(self.grid_size[0], self.grid_size[1], device=self.device)

        # Create the hexagonal coordinates and weights
        self.hex_coords = self._create_hex_grid()
        self.weights = self._initialize_weights()

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

    def _initialize_weights(self):
        """Initialize weights, optionally using PCA, with tensors on the specified device."""
        if self.init_pca and self.data_for_init is not None:
            print("Initializing weights using PCA...")
            pca = PCA(n_components=self.input_dim)
            pca.fit(self.data_for_init)
            components = torch.tensor(pca.components_, dtype=torch.float32).to(self.device)

            # Initialize weights by projecting hexagonal coordinates onto principal components
            # and adding small noise
            rows, cols = self.grid_size
            normalized_hex_coords = self.hex_coords.clone()
            normalized_hex_coords[:, :, 0] /= (cols + 0.5)
            normalized_hex_coords[:, :, 1] /= (rows * 0.866)
            normalized_hex_coords = normalized_hex_coords.view(-1, 2)

            projection = torch.matmul(normalized_hex_coords, components[:2, :].T) # Use first 2 PCs
            weights = projection.view(rows, cols, self.input_dim) + torch.randn(rows, cols, self.input_dim, device=self.device) * 0.01
            print("PCA initialization complete.")
            return weights
        else:
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

    def find_bmu(self, x, winner_penalty=0.0, entropy_penalty=0.0, temperature=1e-8):
        """
        Find the Best Matching Units (BMUs) for each sample in x.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D).
            winner_penalty (float): Penalty on frequently winning neurons.
            entropy_penalty (float): Encourages usage of underutilized neurons.
            temperature (float): Softmax temperature for sampling BMUs.

        Returns:
            (bmu_rows, bmu_cols): Tensors of shape (B,) with BMU coordinates.
        """
        distances = self.forward(x)  # (B, rows, cols)

        if winner_penalty > 0.0 or entropy_penalty > 0.0:
            usage = self.bmu_counts / (self.bmu_counts.sum() + 1e-8)  # (rows, cols)

            penalty = torch.zeros_like(usage)

            if winner_penalty > 0.0:
                penalty += winner_penalty * (usage ** 2)

            if entropy_penalty > 0.0:
                scale_factor = 1
                entropy_bonus = torch.exp(-usage * scale_factor)  # e.g., scale_factor = 10.0
                penalty += entropy_penalty * entropy_bonus

            distances = distances + penalty[None, :, :]  # Broadcast over batch

        flat_distances = distances.view(x.size(0), -1)
        soft_probs = torch.softmax(-flat_distances / temperature, dim=1)
        sampled_indices = torch.multinomial(soft_probs, 1).squeeze(1)
        bmu_rows = sampled_indices // self.grid_size[1]
        bmu_cols = sampled_indices % self.grid_size[1]

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

    def update_weights(self, x, bmu_rows, bmu_cols, winner_penalty=0.0, entropy_penalty=0.0):
        """Update SOM weights using toroidal distances, normalized neighborhoods, and penalties."""
        batch_size = bmu_rows.size(0)
        rows, cols = self.grid_size
        H, W = rows, cols

        # Get toroidal neighborhood and distances
        neighborhood_mask, row_diff, col_diff = self._get_neighbors_toroidal(bmu_rows, bmu_cols, self.sigma)

        # Hexagonal Gaussian distances
        dist_sq = (col_diff + 0.5 * (row_diff % 2))**2 + (row_diff * 0.866)**2
        logits = -dist_sq / (2 * self.sigma**2)

        # Apply neighborhood mask and Gaussian kernel
        neighborhood = torch.exp(logits) * neighborhood_mask  # (B, H, W)

        # ----- Penalty scaling -----
        usage = self.bmu_counts / (self.bmu_counts.sum() + 1e-8)

        penalty_scale = torch.ones_like(usage, dtype=torch.float32)  # (H, W)

        if winner_penalty > 0.0:
            winner_p = (usage ** 2)
            penalty_scale /= (1.0 + winner_penalty * winner_p)

        if entropy_penalty > 0.0:
            entropy_bonus = 1.0 / (usage + 1e-6)
            entropy_bonus = entropy_bonus / entropy_bonus.max()
            penalty_scale *= (1.0 + entropy_penalty * entropy_bonus)

        penalty_scale = penalty_scale.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

        # ----- Compute update -----
        x_expanded = x[:, None, None, :]                     # (B, 1, 1, D)
        weights = self.weights.unsqueeze(0)                  # (1, H, W, D)
        neighborhood_expanded = neighborhood.unsqueeze(-1)   # (B, H, W, 1)

        delta = self.lr * neighborhood_expanded * penalty_scale * (x_expanded - weights)

        # Average over batch
        weight_update = torch.sum(delta, dim=0) / batch_size
        self.weights += weight_update

    def sample_posterior(self, batch_means, batch_logvars):
        # Add noise equivelant to magnitude of logvar to estimate uncertainty
        std = torch.exp(0.5 * batch_logvars)  # Standard deviation
        eps = torch.randn_like(std)     # Sample from a standard normal distribution
        z = batch_means + eps * std            # Reparameterized sample
        return z

    def train(self, means_in, logvars_in, num_epochs, savedir):
        """Train the SOM on the input data."""
        means_tensor = torch.tensor(means_in, dtype=torch.float32, device=self.device)
        logvars_tensor = torch.tensor(logvars_in, dtype=torch.float32, device=self.device)
        num_samples = means_tensor.shape[0]
        indices = torch.arange(num_samples, device=self.device)

        # logger = SOMLogger(self, savedir=savedir, gpu_id=self.device)

        if self.init_pca and self.data_for_init is None:
            self.data_for_init = means_in
            self.weights = self._initialize_weights()

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
                bmu_rows, bmu_cols = self.find_bmu(batch_sampled, self.winner_penalty, self.entropy_penalty, self.temperature)

                # Update winner counts 
                for r, c in zip(bmu_rows, bmu_cols):
                    self.bmu_counts[r, c] += 1

                # Update weights
                self.update_weights(batch_sampled, bmu_rows, bmu_cols, self.winner_penalty)

                # Progress tracking
                batch_num = batch_start // self.batch_size
                total_batches = math.ceil(num_samples / self.batch_size)
                print(f"\rToroidal SOM Epoch: {epoch}/{num_epochs-1}, "
                      f"Batch: {batch_num}/{total_batches} [Batch Size: {self.batch_size}], "
                      f"Iter:{batch_start}/{shuffled_indices.shape[0]-1}, "
                      f"Sigma: {self.sigma:.4f}, LR: {self.lr:.6f}", end="")

            # Decay learning rate and sigma
            self.lr *= self.lr_epoch_decay
            self.bmu_counts *= self.bmu_count_decay
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



