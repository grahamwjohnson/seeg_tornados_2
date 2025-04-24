import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import math

class SphericalSOM(nn.Module):
    """
    Optimized Spherical Self-Organizing Map (SOM) with hexagonal geometry in PyTorch.
    (Includes explicit device placement)
    """

    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma,
                 sigma_epoch_decay, sigma_min, device, init_pca=False, data_for_pca=None):
        super(SphericalSOM, self).__init__()
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

        # Create the spherical coordinates and weights
        self.hex_coords, self.sphere_coords = self._create_spherical_hex_grid()
        self.weights = self._initialize_weights()

    def _create_spherical_hex_grid(self):
        """
        Create a hexagonal grid mapped onto a sphere.
        Returns both the original hexagonal coordinates and their spherical mapping as PyTorch tensors on the specified device.
        """
        rows, cols = self.grid_size

        # Adjust hexagonal coordinates
        hex_coords = torch.zeros(rows, cols, 2, device="cpu")  # Initialize on CPU for easier indexing
        for i in range(rows):
            for j in range(cols):
                x = j + (0.5 * (i % 2))
                y = i * 0.866
                hex_coords[i, j, 0] = x
                hex_coords[i, j, 1] = y
        hex_coords = hex_coords.to(self.device)

        # Normalize hexagonal coordinates to [0, 1] range
        hex_coords[:, :, 0] /= (cols + 0.5)
        hex_coords[:, :, 1] /= (rows * 0.866)

        # Map to sphere using equal-area mapping (UV mapping)
        sphere_coords = torch.zeros(rows, cols, 3, device="cpu") # Initialize on CPU
        for i in range(rows):
            for j in range(cols):
                lon = 2 * math.pi * hex_coords[i, j, 0] - math.pi
                lat = math.pi * hex_coords[i, j, 1] - math.pi/2
                sphere_coords[i, j, 0] = torch.cos(lat) * torch.cos(lon)
                sphere_coords[i, j, 1] = torch.cos(lat) * torch.sin(lon)
                sphere_coords[i, j, 2] = torch.sin(lat)
        sphere_coords = sphere_coords.to(self.device)

        return hex_coords, sphere_coords

    def _initialize_weights(self):
        """Initialize weights, optionally using PCA, with tensors on the specified device."""
        if self.init_pca and self.data_for_pca is not None:
            print("Initializing weights using PCA...")
            pca = PCA(n_components=min(self.input_dim, 3))
            pca.fit(self.data_for_pca)
            components = torch.tensor(pca.components_, dtype=torch.float32).to(self.device) # Move components to device

            # Initialize weights based on PCA components and sphere coordinates
            weights = torch.zeros(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)
            n_components = min(3, components.shape[0])
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    sphere_vec = self.sphere_coords[i, j, :n_components]
                    projection = torch.matmul(sphere_vec, components[:n_components, :])
                    weights[i, j, :] = projection + torch.randn(self.input_dim, device=self.device) * 0.01
            print("PCA initialization complete.")
            return weights
        else:
            print("Initializing weights randomly...")
            return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)

    def reset_device(self, device):
        """Move all tensors to the specified device."""
        self.hex_coords = self.hex_coords.to(device)
        self.sphere_coords = self.sphere_coords.to(device)
        self.weights = self.weights.to(device)
        self.device = device

    def forward(self, x):
        """Compute distances from input x to all SOM neurons."""
        x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
        distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)  # Squared Euclidean distance
        return distances

    def find_bmu(self, x):
        """Find Best Matching Units (BMUs) for each input vector."""
        distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
        flat_distances = distances.view(x.size(0), -1)
        bmu_flat_indices = torch.argmin(flat_distances, dim=1)
        bmu_rows = bmu_flat_indices // self.grid_size[1]
        bmu_cols = bmu_flat_indices % self.grid_size[1]
        return bmu_rows, bmu_cols

    def compute_sphere_distances(self, bmu_rows, bmu_cols):
        """
        Compute great-circle distances on the sphere between BMUs and all neurons using vectorized operations.
        Uses the arccos of the dot product of the normalized vectors.
        """
        batch_size = bmu_rows.size(0)
        grid_rows, grid_cols, _ = self.sphere_coords.shape

        # Create indices for batch, rows, and cols
        batch_indices = torch.arange(batch_size, device=self.device)

        # Gather BMU sphere coordinates using advanced indexing
        bmu_sphere_coords = self.sphere_coords[bmu_rows, bmu_cols]  # Shape: (batch_size, 3)
        bmu_sphere_coords_expanded = bmu_sphere_coords[:, None, None, :]  # Shape: (batch_size, 1, 1, 3)

        # Reshape sphere coordinates for broadcasting
        all_sphere_coords = self.sphere_coords.unsqueeze(0)  # Shape: (1, grid_rows, grid_cols, 3)

        # Calculate the dot product between unit vectors (cos of the angle)
        dot_products = torch.sum(bmu_sphere_coords_expanded * all_sphere_coords, dim=3)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)

        # Convert to angle (great-circle distance on unit sphere)
        sphere_distances = torch.acos(dot_products)  # Shape: (batch_size, grid_rows, grid_cols)

        return sphere_distances

    # def update_weights(self, x, bmu_rows, bmu_cols):
    #     """Update SOM weights using spherical distances and vectorized operations."""
    #     # Compute spherical distances between BMUs and all neurons
    #     sphere_distances = self.compute_sphere_distances(bmu_rows, bmu_cols)

    #     # Compute Gaussian neighborhood function (uses spherical distances)
    #     # The sigma is interpreted as radians on the unit sphere
    #     neighborhood = torch.exp(-sphere_distances**2 / (2 * self.sigma**2))

    #     # Compute weight updates using broadcasting
    #     x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
    #     delta = self.lr * neighborhood[:, :, :, None] * (x_expanded - self.weights)

    #     # Average updates over batch for stability
    #     weight_update = delta.mean(dim=0)
    #     self.weights += weight_update

    # def update_weights(self, x, bmu_rows, bmu_cols):
    #     """Update SOM weights using spherical proximity (dot product) for neighborhood."""
    #     batch_size = bmu_rows.size(0)
    #     grid_rows, grid_cols, _ = self.sphere_coords.shape

    #     # Create indices for batch, rows, and cols
    #     batch_indices = torch.arange(batch_size, device=self.device)

    #     # Gather BMU sphere coordinates
    #     bmu_sphere_coords = self.sphere_coords[bmu_rows, bmu_cols]
    #     bmu_sphere_coords_expanded = bmu_sphere_coords[:, None, None, :]

    #     # Reshape all sphere coordinates for dot product calculation
    #     all_sphere_coords = self.sphere_coords.unsqueeze(0)

    #     # Calculate dot product (similarity)
    #     dot_products = torch.sum(bmu_sphere_coords_expanded * all_sphere_coords, dim=3)

    #     # Define neighborhood based on dot product
    #     # A higher dot product (closer) should give a higher neighborhood strength
    #     # We can use a function of the dot product, for example, a Gaussian centered at 1 (max dot product)
    #     # Or a simple linear scaling if the dot product is normalized.

    #     # Option 1: Gaussian centered at 1
    #     similarity = (dot_products + 1) / 2.0  # Normalize dot product to [0, 1]
    #     neighborhood = torch.exp(-(1 - similarity)**2 / (2 * (self.sigma / 2)**2)) # Adjust sigma scale

    #     # Option 2: Linear scaling (simpler)
    #     # neighborhood = torch.clamp(dot_products, 0.0, 1.0) # Ensure non-negative

    #     # Compute weight updates using the new neighborhood
    #     x_expanded = x[:, None, None, :]
    #     delta = self.lr * neighborhood[:, :, :, None] * (x_expanded - self.weights)

    #     # Average updates over batch
    #     weight_update = delta.mean(dim=0)
    #     self.weights += weight_update

    def update_weights(self, x, bmu_rows, bmu_cols):
        """Update SOM weights with row-dependent learning rate."""
        batch_size = bmu_rows.size(0)
        grid_rows, grid_cols, _ = self.sphere_coords.shape

        # Compute spherical distances
        sphere_distances = self.compute_sphere_distances(bmu_rows, bmu_cols)

        # Compute Gaussian neighborhood function
        neighborhood = torch.exp(-sphere_distances**2 / (2 * self.sigma**2))

        x_expanded = x[:, None, None, :]

        weight_update = torch.zeros_like(self.weights)
        for b in range(batch_size):
            bmu_row = bmu_rows[b].item()
            # Define a learning rate modifier based on distance from the poles
            lr_multiplier = 1.0 - min(bmu_row, grid_rows - 1 - bmu_row) / (grid_rows / 2)
            lr_effective = self.lr * lr_multiplier

            # Calculate the weight change for this batch element
            delta = lr_effective * neighborhood[b, :, :, None] * (x_expanded[b] - self.weights)
            weight_update += delta

        # Average the weight updates over the batch
        self.weights += weight_update / batch_size

    def train(self, data_in, num_epochs):
        """Train the SOM on the input data."""
        data_tensor = torch.tensor(data_in, dtype=torch.float32, device=self.device)

        # If PCA initialization was specified but data wasn't provided during init
        if self.init_pca and self.data_for_pca is None:
            self.data_for_pca = data_in
            self.weights = self._initialize_weights()

        total_batches = (data_tensor.shape[0] + self.batch_size - 1) // self.batch_size

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(data_tensor.shape[0], device=self.device)
            shuffled_data = data_tensor[indices]

            # Train in batches
            for batch_idx in range(0, data_tensor.shape[0], self.batch_size):
                batch = shuffled_data[batch_idx:batch_idx + self.batch_size]

                # Find BMUs
                bmu_rows, bmu_cols = self.find_bmu(batch)

                # Update weights
                self.update_weights(batch, bmu_rows, bmu_cols)

                # Progress tracking
                current_batch = batch_idx // self.batch_size + 1
                print(f"\rSpherical SOM Epoch: {epoch+1}/{num_epochs}, "
                      f"Batch: {current_batch}/{total_batches}, "
                      f"Iter:{batch_idx}/{shuffled_data.shape[0]-1}, "
                      f"Sigma: {self.sigma:.4f}, LR: {self.lr:.6f}", end="")
                
            # Decay learning rate and sigma
            self.lr *= self.lr_epoch_decay
            self.sigma = max(self.sigma * self.sigma_epoch_decay, self.sigma_min)

    def get_weights(self):
        """Return weights as a NumPy array (for visualization)."""
        return self.weights.cpu().numpy()

    def get_sphere_coords(self):
        """Return the spherical coordinates of all neurons as a NumPy array."""
        return self.sphere_coords.cpu().numpy()

    def get_hex_coords(self):
        """Return the 2D hexagonal coordinates as a NumPy array."""
        return self.hex_coords.cpu().numpy()

    def project_data(self, data):
        """Project data onto the SOM grid and return grid positions."""
        data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        bmu_rows = []
        bmu_cols = []

        # Process in batches to avoid memory issues
        for i in range(0, data_tensor.shape[0], self.batch_size):
            batch = data_tensor[i:i + self.batch_size]
            batch_rows, batch_cols = self.find_bmu(batch)
            bmu_rows.append(batch_rows.cpu().numpy())
            bmu_cols.append(batch_cols.cpu().numpy())

        return np.concatenate(bmu_rows), np.concatenate(bmu_cols)
    
# # class SphericalSOM(nn.Module):
#     """
#     Spherical Self-Organizing Map (SOM) with hexagonal geometry implemented in PyTorch.
    
#     This class implements a Self-Organizing Map that wraps a 2D grid onto a sphere,
#     eliminating edge effects, and uses hexagonal geometry for better neighborhood
#     representation. Each point on the grid is mapped to the surface of a unit sphere,
#     and distances between neurons are computed along the surface of the sphere.
    
#     Key Features:
#     - Spherical Topology: Maps neurons onto a sphere to eliminate edge effects
#     - Hexagonal Geometry: Uses hexagonal grid for better neighborhood representation
#     - Batch Training: Supports efficient GPU-accelerated batch training
#     - Dynamically Decaying Parameters: Learning rate and neighborhood radius decay over time
#     - PCA-based Initialization: Optional weight initialization using PCA components
    
#     Parameters:
#     - grid_size (tuple): The size of the SOM grid, specified as (rows, cols)
#     - input_dim (int): The dimensionality of the input data
#     - batch_size (int): The number of input samples processed in a single batch
#     - lr (float): The initial learning rate for weight updates
#     - lr_epoch_decay (float): The decay factor for the learning rate after each epoch
#     - sigma (float): The initial radius of the neighborhood function
#     - sigma_epoch_decay (float): The decay factor for sigma over epochs
#     - sigma_min (float): The minimum value of sigma
#     - device (str or torch.device): The device to run the model on
#     - init_pca (bool): Whether to initialize weights using PCA components
#     - data_for_pca (numpy.ndarray): Optional data for PCA initialization
#     """

#     def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, 
#                  sigma_epoch_decay, sigma_min, device, init_pca=False, data_for_pca=None):
#         super(SphericalSOM, self).__init__()
#         self.grid_size = grid_size
#         self.input_dim = input_dim
#         self.lr = lr
#         self.lr_epoch_decay = lr_epoch_decay
#         self.sigma = sigma
#         self.sigma_epoch_decay = sigma_epoch_decay
#         self.sigma_min = sigma_min
#         self.batch_size = batch_size
#         self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
#         self.init_pca = init_pca
#         self.data_for_pca = data_for_pca
        
#         # Create the spherical coordinates and weights
#         self.hex_coords, self.sphere_coords = self._create_spherical_hex_grid()
#         self.weights = self._initialize_weights()
        
#     def _create_spherical_hex_grid(self):
#         """
#         Create a hexagonal grid mapped onto a sphere.
#         Returns both the original hexagonal coordinates and their spherical mapping.
#         """
#         # Create a rectangular grid for mapping
#         rows, cols = self.grid_size
        
#         # Adjust hexagonal coordinates
#         hex_coords = torch.zeros(rows, cols, 2, device=self.device)
        
#         for i in range(rows):
#             for j in range(cols):
#                 # In a hexagonal grid, odd rows are offset by 0.5
#                 x = j + (0.5 * (i % 2))
#                 y = i * 0.866  # sqrt(3)/2 = 0.866, hexagon height
#                 hex_coords[i, j, 0] = x
#                 hex_coords[i, j, 1] = y
        
#         # Normalize hexagonal coordinates to [0, 1] range
#         hex_coords[:, :, 0] /= cols + 0.5
#         hex_coords[:, :, 1] /= rows * 0.866
        
#         # Map to sphere using equal-area mapping (UV mapping)
#         sphere_coords = torch.zeros(rows, cols, 3, device=self.device)
        
#         # Convert from [0,1]x[0,1] to spherical coordinates
#         for i in range(rows):
#             for j in range(cols):
#                 # Map to latitude and longitude
#                 lon = 2 * math.pi * hex_coords[i, j, 0] - math.pi  # longitude: -π to π
#                 lat = math.pi * hex_coords[i, j, 1] - math.pi/2    # latitude: -π/2 to π/2
                
#                 # Convert to 3D Cartesian coordinates on unit sphere
#                 sphere_coords[i, j, 0] = torch.cos(lat) * torch.cos(lon)  # x
#                 sphere_coords[i, j, 1] = torch.cos(lat) * torch.sin(lon)  # y
#                 sphere_coords[i, j, 2] = torch.sin(lat)                   # z
                
#         return hex_coords, sphere_coords
        
#     def _initialize_weights(self):
#         """Initialize weights, optionally using PCA."""
#         if self.init_pca and self.data_for_pca is not None:
#             print("Initializing weights using PCA...")
#             pca = PCA(n_components=min(self.input_dim, 3))
#             pca.fit(self.data_for_pca)
#             components = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)
            
#             # Initialize weights based on PCA components and sphere coordinates
#             weights = torch.zeros(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)
            
#             # Use the first 3 PCs (or fewer if input_dim < 3) to project the sphere coordinates
#             n_components = min(3, components.shape[0])
#             for i in range(self.grid_size[0]):
#                 for j in range(self.grid_size[1]):
#                     # Project the sphere coordinates onto principal components
#                     sphere_vec = self.sphere_coords[i, j, :n_components]
#                     projection = torch.matmul(sphere_vec, components[:n_components, :])
                    
#                     # Add small random noise
#                     weights[i, j, :] = projection + torch.randn(self.input_dim, device=self.device) * 0.01
                    
#             print("PCA initialization complete.")
#             return weights
#         else:
#             print("Initializing weights randomly...")
#             return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)
        
#     def reset_device(self, device):
#         """Move all tensors to the specified device."""
#         new_device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
#         self.hex_coords = self.hex_coords.to(new_device)
#         self.sphere_coords = self.sphere_coords.to(new_device)
#         self.weights = self.weights.to(new_device)
#         self.device = new_device
        
#     def forward(self, x):
#         """Compute distances from input x to all SOM neurons."""
#         x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
#         distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)  # Squared Euclidean distance
#         return distances
        
#     def find_bmu(self, x):
#         """Find Best Matching Units (BMUs) for each input vector."""
#         distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
        
#         # Flatten the grid and find BMU indices
#         flat_distances = distances.view(x.size(0), -1)
#         bmu_flat_indices = torch.argmin(flat_distances, dim=1)
        
#         # Convert flat indices to 2D grid coordinates
#         bmu_rows = bmu_flat_indices // self.grid_size[1]
#         bmu_cols = bmu_flat_indices % self.grid_size[1]
        
#         return bmu_rows, bmu_cols
        
#     def compute_sphere_distances(self, bmu_rows, bmu_cols):
#         """
#         Compute great-circle distances on the sphere between BMUs and all neurons.
#         Uses the haversine formula for numerical stability.
#         """
#         batch_size = bmu_rows.size(0)
        
#         # Get the BMU sphere coordinates
#         bmu_sphere_coords = torch.zeros(batch_size, 3, device=self.device)
#         for b in range(batch_size):
#             bmu_sphere_coords[b] = self.sphere_coords[bmu_rows[b], bmu_cols[b]]
        
#         # Reshape for broadcasting
#         bmu_coords_expanded = bmu_sphere_coords.view(batch_size, 1, 1, 3)
        
#         # Calculate the dot product between unit vectors (cos of the angle)
#         # Clip values to ensure numerical stability
#         dot_products = torch.sum(bmu_coords_expanded * self.sphere_coords.unsqueeze(0), dim=3)
#         dot_products = torch.clamp(dot_products, -1.0, 1.0)
        
#         # Convert to angle (great-circle distance on unit sphere)
#         # Since we're on a unit sphere, the angle equals the distance
#         sphere_distances = torch.acos(dot_products)
        
#         return sphere_distances
        
#     def update_weights(self, x, bmu_rows, bmu_cols):
#         """Update SOM weights using spherical distances and vectorized operations."""
#         # Compute spherical distances between BMUs and all neurons
#         sphere_distances = self.compute_sphere_distances(bmu_rows, bmu_cols)
        
#         # Compute Gaussian neighborhood function (uses spherical distances)
#         # The sigma is interpreted as radians on the unit sphere
#         neighborhood = torch.exp(-sphere_distances**2 / (2 * self.sigma**2))
        
#         # Compute weight updates using broadcasting
#         x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
#         delta = self.lr * neighborhood[:, :, :, None] * (x_expanded - self.weights)
        
#         # Average updates over batch for stability
#         weight_update = delta.mean(dim=0)
#         self.weights += weight_update
        
#     def train(self, data_in, num_epochs):
#         """Train the SOM on the input data."""
#         data_tensor = torch.tensor(data_in, dtype=torch.float32, device=self.device)
        
#         # If PCA initialization was specified but data wasn't provided during init
#         if self.init_pca and self.data_for_pca is None:
#             self.data_for_pca = data_in
#             self.weights = self._initialize_weights()
            
#         total_batches = (data_tensor.shape[0] + self.batch_size - 1) // self.batch_size
            
#         for epoch in range(num_epochs):
#             # Shuffle data
#             indices = torch.randperm(data_tensor.shape[0], device=self.device)
#             shuffled_data = data_tensor[indices]
            
#             # Train in batches
#             for batch_idx in range(0, data_tensor.shape[0], self.batch_size):
#                 batch = shuffled_data[batch_idx:batch_idx + self.batch_size]
                
#                 # Find BMUs
#                 bmu_rows, bmu_cols = self.find_bmu(batch)
                
#                 # Update weights
#                 self.update_weights(batch, bmu_rows, bmu_cols)
                
#                 # Progress tracking
#                 current_batch = batch_idx // self.batch_size + 1
#                 print(f"\rSpherical SOM Epoch: {epoch+1}/{num_epochs}, "
#                       f"Batch: {current_batch}/{total_batches}, "
#                       f"Sigma: {self.sigma:.4f}, LR: {self.lr:.6f}", end="")
            
#             print()  # New line after each epoch
            
#             # Decay learning rate and sigma
#             self.lr *= self.lr_epoch_decay
#             self.sigma = max(self.sigma * self.sigma_epoch_decay, self.sigma_min)
            
#     def get_weights(self):
#         """Return weights as a NumPy array (for visualization)."""
#         return self.weights.cpu().numpy()
        
#     def get_sphere_coords(self):
#         """Return the spherical coordinates of all neurons as a NumPy array."""
#         return self.sphere_coords.cpu().numpy()
        
#     def get_hex_coords(self):
#         """Return the 2D hexagonal coordinates as a NumPy array."""
#         return self.hex_coords.cpu().numpy()
        
#     def project_data(self, data):
        """Project data onto the SOM grid and return grid positions."""
        data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        bmu_rows = []
        bmu_cols = []
        
        # Process in batches to avoid memory issues
        for i in range(0, data_tensor.shape[0], self.batch_size):
            batch = data_tensor[i:i + self.batch_size]
            batch_rows, batch_cols = self.find_bmu(batch)
            bmu_rows.append(batch_rows.cpu().numpy())
            bmu_cols.append(batch_cols.cpu().numpy())
            
        return np.concatenate(bmu_rows), np.concatenate(bmu_cols)