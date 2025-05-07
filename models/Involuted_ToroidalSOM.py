import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances

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


class Involuted_ToroidalSOM(nn.Module):
    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma,
                 sigma_epoch_decay, sigma_min, device, init_pca=False, data_for_pca=None, **kwargs):
        super(Involuted_ToroidalSOM, self).__init__()
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

        self.hex_coords = self._create_hex_grid()
        self.weights = self._initialize_weights()
        self.bmu_usage = torch.zeros(self.grid_size, dtype=torch.float32, device=self.device)

    def _create_hex_grid(self):
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
        if self.init_pca and self.data_for_pca is not None:
            print("Initializing weights using PCA...")
            pca = PCA(n_components=self.input_dim)
            pca.fit(self.data_for_pca)
            components = torch.tensor(pca.components_, dtype=torch.float32).to(self.device)

            rows, cols = self.grid_size
            normalized_hex_coords = self.hex_coords.clone()
            normalized_hex_coords[:, :, 0] /= (cols + 0.5)
            normalized_hex_coords[:, :, 1] /= (rows * 0.866)
            normalized_hex_coords = normalized_hex_coords.view(-1, 2)

            projection = torch.matmul(normalized_hex_coords, components[:2, :].T)
            weights = projection.view(rows, cols, self.input_dim) + torch.randn(rows, cols, self.input_dim, device=self.device) * 0.01
            print("PCA initialization complete.")
            return weights
        else:
            print("Initializing weights randomly...")
            return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)

    def reset_device(self, device):
        self.hex_coords = self.hex_coords.to(device)
        self.weights = self.weights.to(device)
        self.bmu_usage = self.bmu_usage.to(device)
        self.device = device

    def forward(self, x):
        x_expanded = x[:, None, None, :]
        distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)
        return distances

    def find_bmu(self, x):
        distances = self.forward(x)
        flat_distances = distances.view(x.size(0), -1)
        bmu_flat_indices = torch.argmin(flat_distances, dim=1)
        bmu_rows = bmu_flat_indices // self.grid_size[1]
        bmu_cols = bmu_flat_indices % self.grid_size[1]
        return bmu_rows, bmu_cols

    def _get_neighbors_toroidal(self, bmu_rows, bmu_cols):
        batch_size = bmu_rows.size(0)
        rows, cols = self.grid_size

        row_indices = torch.arange(rows, device=self.device).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, cols)
        col_indices = torch.arange(cols, device=self.device).float().unsqueeze(0).unsqueeze(0).repeat(batch_size, rows, 1)

        bmu_row_expanded = bmu_rows.float().unsqueeze(-1).unsqueeze(-1).repeat(1, rows, cols)
        bmu_col_expanded = bmu_cols.float().unsqueeze(-1).unsqueeze(-1).repeat(1, rows, cols)

        row_diff = torch.abs(row_indices - bmu_row_expanded)
        col_diff = torch.abs(col_indices - bmu_col_expanded)

        row_diff = torch.min(row_diff, rows - row_diff)
        col_diff = torch.min(col_diff, cols - col_diff)

        dist_sq = (col_diff + 0.5 * (row_diff % 2))**2 + (row_diff * 0.866)**2
        max_dist_sq = (self.sigma * 1.5)**2
        neighborhood_mask = (dist_sq < max_dist_sq).float()

        return neighborhood_mask, row_diff, col_diff

    def _involuted_clamped_bump(self, x, y, A, w=2.0, S=1.0):
        r = torch.sqrt(x**2 + y**2)
        r_scaled = r / S
        R = 2 * A
        center = A * torch.exp(-r_scaled**2)
        ring = torch.exp(-((r_scaled - R)**2) / (w**2))
        z = ring - center
        return torch.clamp(z, min=0.0)

    def update_weights(self, x, bmu_rows, bmu_cols):
        batch_size = bmu_rows.size(0)
        rows, cols = self.grid_size

        neighborhood_mask, row_diff, col_diff = self._get_neighbors_toroidal(bmu_rows, bmu_cols)

        # Compute usage-scaled A for each sample
        with torch.no_grad():
            A = self.bmu_usage / (self.bmu_usage.max() + 1e-8)
        A_batch = A[bmu_rows, bmu_cols].view(-1, 1, 1).expand(-1, rows, cols)

        # Apply custom kernel
        neighborhood = self._involuted_clamped_bump(col_diff, row_diff, A_batch, w=2.0, S=self.sigma)
        neighborhood *= neighborhood_mask

        x_expanded = x[:, None, None, :]
        neighborhood_expanded = neighborhood.unsqueeze(-1)

        delta = self.lr * neighborhood_expanded * (x_expanded - self.weights)
        weight_update = torch.sum(delta, dim=0) / batch_size
        self.weights += weight_update

    def sample_posterior(self, batch_means, batch_logvars):
        std = torch.exp(0.5 * batch_logvars)
        eps = torch.randn_like(std)
        z = batch_means + eps * std
        return z

    def train(self, means_in, logvars_in, num_epochs):
        means_tensor = torch.tensor(means_in, dtype=torch.float32, device=self.device)
        logvars_tensor = torch.tensor(logvars_in, dtype=torch.float32, device=self.device)
        num_samples = means_tensor.shape[0]
        indices = torch.arange(num_samples, device=self.device)

        if self.init_pca and self.data_for_pca is None:
            self.data_for_pca = means_in
            self.weights = self._initialize_weights()

        for epoch in range(num_epochs):
            shuffled_indices = indices[torch.randperm(num_samples)]

            for batch_start in range(0, num_samples, self.batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + self.batch_size]
                batch_means = means_tensor[batch_indices]
                batch_logvars = logvars_tensor[batch_indices]
                batch_sampled = self.sample_posterior(batch_means, batch_logvars)

                bmu_rows, bmu_cols = self.find_bmu(batch_sampled)

                # Track BMU usage
                with torch.no_grad():
                    for r, c in zip(bmu_rows, bmu_cols):
                        self.bmu_usage[r, c] += 1

                self.update_weights(batch_sampled, bmu_rows, bmu_cols)

                batch_num = batch_start // self.batch_size
                total_batches = math.ceil(num_samples / self.batch_size)
                print(f"\rToroidal SOM Epoch: {epoch}/{num_epochs-1}, "
                      f"Batch: {batch_num}/{total_batches} [Batch Size: {self.batch_size}], "
                      f"Iter:{batch_start}/{shuffled_indices.shape[0]-1}, "
                      f"Sigma: {self.sigma:.4f}, LR: {self.lr:.6f}", end="")

            self.lr *= self.lr_epoch_decay
            self.sigma = max(self.sigma * self.sigma_epoch_decay, self.sigma_min)

    def get_weights(self):
        return self.weights.cpu().numpy()

    def get_hex_coords(self):
        return self.hex_coords.cpu().numpy()

    def project_data(self, data):
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        bmu_rows, bmu_cols = self.find_bmu(data)
        return bmu_rows.cpu().numpy(), bmu_cols.cpu().numpy()
