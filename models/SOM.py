import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class SOM(nn.Module):
    """
    Self-Organizing Map (SOM) implementation in PyTorch.

    This class defines a Self-Organizing Map (SOM) for unsupervised learning. SOM is a 
    type of artificial neural network that performs dimensionality reduction through 
    clustering of data points in a lower-dimensional grid. This implementation supports 
    the training of a 2D grid of neurons, where each neuron represents a weight vector, 
    and the network learns to represent the input data in an organized fashion. The neurons 
    are updated using a competitive learning process where the "Best Matching Unit" (BMU) is 
    found for each input sample, and the weights of the BMU and its neighbors are updated 
    according to a Gaussian neighborhood function.

    ### Key Features:
    - **Batch Training:** Supports efficient batch training of the SOM with the ability 
    to update weights using vectorized operations for parallel processing.
    - **Learning Rate Decay:** The learning rate (`lr`) decays over epochs according to 
    a specified decay factor (`lr_epoch_decay`).
    - **Neighborhood Function:** The weight update is influenced by a Gaussian neighborhood 
    function, where the neighborhood size decays over time with a specified `sigma`.
    - **Device Compatibility:** The model is designed to work on both CPU and GPU, with 
    seamless device management.
    - **Flexible Initialization:** Weights are initialized randomly and can be updated 
    based on input data.

    ### Parameters:
    - `grid_size` (tuple): The size of the SOM grid, specified as (rows, cols).
    - `input_dim` (int): The dimensionality of the input data (e.g., number of features 
    in each input vector).
    - `batch_size` (int): The number of input samples processed in a single batch during 
    training.
    - `lr` (float): The initial learning rate for weight updates.
    - `lr_epoch_decay` (float): The decay factor for the learning rate after each epoch.
    - `sigma` (float): The initial radius of the neighborhood function used to update weights.
    - `sigma_epoch_decay` (float): The decay factor for `sigma` over epochs.
    - `sigma_min` (float): The minimum value of `sigma` to prevent it from decaying too much.
    - `device` (str or torch.device): The device to run the model on (e.g., "cpu" or "cuda").

    ### Methods:
    1. **`__init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, sigma_epoch_decay, sigma_min, device)`**: 
    Initializes the SOM model with the specified parameters.
    2. **`reset_device(self, device)`**: Resets the model's device (e.g., to switch between 
    CPU and GPU) and moves the weights and coordinates to the new device.
    3. **`forward(self, x)`**: Computes the squared Euclidean distance between the input 
    `x` and all the neurons in the SOM grid.
    4. **`find_bmu(self, x)`**: Finds the Best Matching Unit (BMU) for a given input `x` 
    by computing the distances to all neurons and selecting the one with the smallest 
    distance.
    5. **`update_weights(self, x, bmu_rows, bmu_cols)`**: Updates the weights of the SOM 
    neurons using the BMU and a Gaussian neighborhood function. This method uses 
    efficient vectorized operations for weight updates.
    6. **`train(self, data_in, num_epochs)`**: Trains the SOM on the input data for a 
    specified number of epochs. The method shuffles the data, finds the BMU for each 
    batch, and updates the weights accordingly.
    7. **`get_weights(self)`**: Returns the current weights of the SOM as a NumPy array 
    for visualization purposes.

    ### Usage Example:
    ```python
    # Initialize the SOM model
    som = SOM(grid_size=(10, 10), input_dim=3, batch_size=32, lr=0.1, lr_epoch_decay=0.9, 
            sigma=3.0, sigma_epoch_decay=0.9, sigma_min=0.1, device="cuda")

    # Train the model on input data
    som.train(data_in=my_data, num_epochs=100)

    # Retrieve the trained weights for visualization
    weights = som.get_weights()
    
    """

    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, sigma_epoch_decay, sigma_min, device, init_pca=False, data_for_pca=None):
        super(SOM, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = lr
        self.lr_epoch_decay = lr_epoch_decay
        self.sigma = sigma
        self.sigma_epoch_decay = sigma_epoch_decay
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.device = device
        self.init_pca = init_pca
        self.data_for_pca = data_for_pca

        # Precompute coordinate grid
        self.coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size[0], device=device),
            torch.arange(grid_size[1], device=device),
            indexing="ij"
        ), dim=-1)  # Shape: (grid_size[0], grid_size[1], 2)

        # Initialize weights
        self.weights = self._initialize_weights()

    def _initialize_weights(self):
        if self.init_pca and self.data_for_pca is not None:
            print("Initializing weights using PCA...")
            pca = PCA(n_components=self.input_dim)
            pca.fit(self.data_for_pca)
            components = torch.tensor(pca.components_, dtype=torch.float32, device=self.device)

            # Initialize weights based on PCA components
            weights = torch.zeros(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)
            grid_vectors = self.coords.float() / torch.tensor(self.grid_size, dtype=torch.float32, device=self.device)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    # A simple way to use PCA: project grid coordinates onto principal components
                    projection = torch.matmul(grid_vectors[i, j], components[:2, :]) # Using first 2 PCs
                    weights[i, j, :] = torch.randn(self.input_dim, device=self.device) * 0.01 + projection.unsqueeze(0) # Add small random noise
            print("PCA initialization complete.")
            return weights
        else:
            print("Initializing weights randomly...")
            return torch.randn(self.grid_size[0], self.grid_size[1], self.input_dim, device=self.device)

    def reset_device(self, device):
        self.coords = self.coords.to(device)
        self.weights = self.weights.to(device)
        self.device = device

    def forward(self, x):
        """ Compute distances from input x to all SOM neurons """
        x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
        distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)  # Squared Euclidean distance
        return distances

    def find_bmu(self, x):
        """ Find Best Matching Units (BMUs) in parallel """
        distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
        bmu_indices = torch.argmin(distances.view(x.size(0), -1), dim=1)  # Flatten grid and find BMU index
        bmu_coords = self.coords.view(-1, 2)[bmu_indices]  # Retrieve (row, col) coordinates
        return bmu_coords[:, 0], bmu_coords[:, 1]  # Return row, col separately

    def update_weights(self, x, bmu_rows, bmu_cols):
        """ Update SOM weights efficiently using vectorized operations """
        # Compute BMU distances (batch_size, grid_size[0], grid_size[1])
        bmu_coords = torch.stack([bmu_rows, bmu_cols], dim=1)[:, None, None, :]  # Shape: (batch_size, 1, 1, 2)
        neuron_dists = torch.sum((self.coords[None, :, :, :] - bmu_coords) ** 2, dim=-1)  # Squared distance

        # Compute Gaussian neighborhood function
        neighborhood = torch.exp(-neuron_dists / (2 * self.sigma ** 2))  # (batch_size, grid_size[0], grid_size[1])

        # Compute weight updates using broadcasting
        delta = self.lr * neighborhood[:, :, :, None] * (x[:, None, None, :] - self.weights)  # Shape: (batch_size, grid_size[0], grid_size[1], input_dim)
        self.weights += delta.mean(dim=0)  # Average over batch to stabilize updates

    def train(self, data_in, num_epochs):
        """ Train the SOM on the input data """
        data_tensor = torch.tensor(data_in, dtype=torch.float32, device=self.device)  # Move ALL data to GPU
        if self.init_pca and self.data_for_pca is None:
            self.data_for_pca = data_in # Use training data for PCA if not provided during init
            self.weights = self._initialize_weights()

        for epoch in range(num_epochs):
            indices = torch.randperm(data_tensor.shape[0], device=self.device)  # Shuffle using PyTorch
            data = data_tensor[indices]

            # Train in batches
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]  # Get batch
                bmu_rows, bmu_cols = self.find_bmu(batch)  # Find BMUs
                self.update_weights(batch, bmu_rows, bmu_cols)  # Efficiently update weights

                print(f"\rSOM Epoch:{epoch}/{num_epochs-1}, Iter:{i}/{data.shape[0]-1}, Sigma:{self.sigma}, LR:{self.lr}     ", end="")

            # Decay learning rate and sigma over time
            self.lr *= self.lr_epoch_decay
            self.sigma *= self.sigma_epoch_decay

            # Low sigma clipping
            if self.sigma < self.sigma_min:
                self.sigma = self.sigma_min

    def get_weights(self):
        """ Return weights as a NumPy array (for visualization) """
        return self.weights.cpu().numpy()

    # def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, sigma_epoch_decay, sigma_min, device):
    #     super(SOM, self).__init__()
    #     self.grid_size = grid_size
    #     self.input_dim = input_dim
    #     self.lr = lr
    #     self.lr_epoch_decay = lr_epoch_decay
    #     self.sigma = sigma
    #     self.sigma_epoch_decay = sigma_epoch_decay
    #     self.sigma_min = sigma_min
    #     self.batch_size = batch_size
    #     self.device = device

    #     # Initialize weights (neurons) and move to GPU
    #     self.weights = torch.randn(grid_size[0], grid_size[1], input_dim, device=self.device)

    #     # Precompute coordinate grid
    #     self.coords = torch.stack(torch.meshgrid(
    #         torch.arange(grid_size[0], device=device),
    #         torch.arange(grid_size[1], device=device),
    #         indexing="ij"
    #     ), dim=-1)  # Shape: (grid_size[0], grid_size[1], 2)

    # def reset_device(self, device):
    #     self.coords = self.coords.to(device)
    #     self.weights = self.weights.to(device)
    #     self.device = device

    # def forward(self, x):
    #     """ Compute distances from input x to all SOM neurons """
    #     x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
    #     distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)  # Squared Euclidean distance
    #     return distances

    # def find_bmu(self, x):
    #     """ Find Best Matching Units (BMUs) in parallel """
    #     distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
    #     bmu_indices = torch.argmin(distances.view(x.size(0), -1), dim=1)  # Flatten grid and find BMU index
    #     bmu_coords = self.coords.view(-1, 2)[bmu_indices]  # Retrieve (row, col) coordinates
    #     return bmu_coords[:, 0], bmu_coords[:, 1]  # Return row, col separately

    # def update_weights(self, x, bmu_rows, bmu_cols):
    #     """ Update SOM weights efficiently using vectorized operations """
    #     # Compute BMU distances (batch_size, grid_size[0], grid_size[1])
    #     bmu_coords = torch.stack([bmu_rows, bmu_cols], dim=1)[:, None, None, :]  # Shape: (batch_size, 1, 1, 2)
    #     neuron_dists = torch.sum((self.coords[None, :, :, :] - bmu_coords) ** 2, dim=-1)  # Squared distance

    #     # Compute Gaussian neighborhood function
    #     neighborhood = torch.exp(-neuron_dists / (2 * self.sigma ** 2))  # (batch_size, grid_size[0], grid_size[1])

    #     # Compute weight updates using broadcasting
    #     delta = self.lr * neighborhood[:, :, :, None] * (x[:, None, None, :] - self.weights)  # Shape: (batch_size, grid_size[0], grid_size[1], input_dim)
    #     self.weights += delta.mean(dim=0)  # Average over batch to stabilize updates

    # def train(self, data_in, num_epochs):
    #     """ Train the SOM on the input data """
    #     data_tensor = torch.tensor(data_in, dtype=torch.float32, device=self.device)  # Move ALL data to GPU
    #     for epoch in range(num_epochs):
    #         indices = torch.randperm(data_tensor.shape[0], device=self.device)  # Shuffle using PyTorch
    #         data = data_tensor[indices]

    #         # Train in batches
    #         for i in range(0, data.shape[0], self.batch_size):
    #             batch = data[i:i + self.batch_size]  # Get batch
    #             bmu_rows, bmu_cols = self.find_bmu(batch)  # Find BMUs
    #             self.update_weights(batch, bmu_rows, bmu_cols)  # Efficiently update weights

    #             print(f"\rSOM Epoch:{epoch}/{num_epochs-1}, Iter:{i}/{data.shape[0]-1}, Sigma:{self.sigma}, LR:{self.lr}     ", end="")

    #         # Decay learning rate and sigma over time
    #         self.lr *= self.lr_epoch_decay
    #         self.sigma *= self.sigma_epoch_decay
            
    #         # Low sigma clipping
    #         if self.sigma < self.sigma_min: 
    #             self.sigma = self.sigma_min

    # def get_weights(self):
    #     """ Return weights as a NumPy array (for visualization) """
    #     return self.weights.cpu().numpy()
