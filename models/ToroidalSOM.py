import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import math

class ToroidalSOM(nn.Module):
    """
    Optimized Toroidal Self-Organizing Map (SOM) with hexagonal geometry in PyTorch.
    Optimized for GPU performance.
    """

    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma,
                 sigma_epoch_decay, sigma_min, device, init_pca=False, data_for_pca=None):
        super(ToroidalSOM, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lr = lr
        self.lr_epoch_decay = lr_epoch_decay
        self.sigma = sigma
        self.sigma_epoch_decay = sigma_epoch_decay
        self.sigma_min = sigma_min
        self.device = device
        self.init_pca = init_pca
        self.data_for_pca = data_for_pca

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
        if self.init_pca and self.data_for_pca is not None:
            print("Initializing weights using PCA...")
            pca = PCA(n_components=self.input_dim)
            pca.fit(self.data_for_pca)
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
            print("Initializing weights randomly...")
            return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)

    def reset_device(self, device):
        """Move all tensors to the specified device."""
        self.hex_coords = self.hex_coords.to(device)
        self.weights = self.weights.to(device)
        self.device = device

    def forward(self, x):
        """Compute squared Euclidean distances from input x to all SOM neurons."""
        x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
        distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)
        return distances

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

        # Compute weight update
        delta = self.lr * neighborhood_expanded * (x_expanded - self.weights)

        # Apply update, averaging over the batch
        weight_update = torch.sum(delta, dim=0) / batch_size
        self.weights += weight_update

    def sample_posterior(self, batch_means, batch_logvars):
        # Add noise equivelant to magnitude of logvar to estimate uncertainty
        std = torch.exp(0.5 * batch_logvars)  # Standard deviation
        eps = torch.randn_like(std)     # Sample from a standard normal distribution
        z = batch_means + eps * std            # Reparameterized sample
        return z

    def train(self, means_in, logvars_in, num_epochs):
        """Train the SOM on the input data."""
        means_tensor = torch.tensor(means_in, dtype=torch.float32, device=self.device)
        logvars_tensor = torch.tensor(logvars_in, dtype=torch.float32, device=self.device)
        num_samples = means_tensor.shape[0]
        indices = torch.arange(num_samples, device=self.device)

        if self.init_pca and self.data_for_pca is None:
            self.data_for_pca = means_in
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
                bmu_rows, bmu_cols = self.find_bmu(batch_sampled)

                # Update weights
                self.update_weights(batch_sampled, bmu_rows, bmu_cols)

                # Progress tracking
                batch_num = batch_start // self.batch_size + 1
                total_batches = math.ceil(num_samples / self.batch_size)
                print(f"\rToroidal SOM Epoch: {epoch+1}/{num_epochs}, "
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