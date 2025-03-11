import torch
import torch.nn as nn

class SOM(nn.Module):
    def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, sigma_epoch_decay, device):
        super(SOM, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = lr
        self.lr_epoch_decay = lr_epoch_decay
        self.sigma = sigma
        self.sigma_epoch_decay = sigma_epoch_decay
        self.batch_size = batch_size
        self.device = device

        # Initialize weights (neurons) and move to GPU
        self.weights = torch.randn(grid_size[0], grid_size[1], input_dim, device=self.device)

        # Precompute coordinate grid
        self.coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size[0], device=device),
            torch.arange(grid_size[1], device=device),
            indexing="ij"
        ), dim=-1)  # Shape: (grid_size[0], grid_size[1], 2)

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
        data_tensor = torch.tensor(data_in, dtype=torch.float32, device=self.device)  # Move data to GPU
        for epoch in range(num_epochs):
            indices = torch.randperm(data_tensor.shape[0], device=self.device)  # Shuffle using PyTorch
            data = data_tensor[indices]

            # Train in batches
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]  # Get batch
                bmu_rows, bmu_cols = self.find_bmu(batch)  # Find BMUs
                self.update_weights(batch, bmu_rows, bmu_cols)  # Efficiently update weights

                print(f"\rSOM Epoch:{epoch}/{num_epochs-1}, Iter:{i}/{data.shape[0]-1}    ", end="")

            # Decay learning rate and sigma over time
            self.lr *= self.lr_epoch_decay
            self.sigma *= self.sigma_epoch_decay

    def get_weights(self):
        """ Return weights as a NumPy array (for visualization) """
        return self.weights.cpu().numpy()



# import torch
# import torch.nn as nn
# import numpy as np
# import sys

# class SOM(nn.Module):
#     def __init__(self, grid_size, input_dim, batch_size, lr, lr_epoch_decay, sigma, sigma_epoch_decay, device):
#         super(SOM, self).__init__()
#         self.grid_size = grid_size
#         self.input_dim = input_dim
#         self.lr = lr
#         self.lr_epoch_decay = lr_epoch_decay
#         self.sigma = sigma
#         self.sigma_epoch_decay = sigma_epoch_decay
#         self.batch_size = batch_size
#         self.device = device

#         # Initialize the weights (neurons) randomly and move to GPU
#         self.weights = torch.randn(grid_size[0], grid_size[1], input_dim, device=self.device)

#     def forward(self, x):
#         # Ensure x has the correct shape
#         if x.dim() == 1:
#             x = x.unsqueeze(0)  # Reshape to (1, input_dim)

#         # Vectorized distance computation
#         x_expanded = x[:, None, None, :]  # Shape: (batch_size, 1, 1, input_dim)
#         distances = torch.sum((self.weights - x_expanded) ** 2, dim=3)  # Shape: (batch_size, grid_size[0], grid_size[1])
#         return distances

#     def find_bmu(self, x):
#         # Find the Best Matching Unit (BMU) for a batch of inputs
#         distances = self.forward(x)  # Shape: (batch_size, grid_size[0], grid_size[1])
#         bmu_indices = torch.argmin(distances.view(x.size(0), -1), dim=1)  # Flatten grid and find BMU
#         bmu_rows = bmu_indices // self.grid_size[1]
#         bmu_cols = bmu_indices % self.grid_size[1]
#         return bmu_rows, bmu_cols

#     def update_weights(self, x, bmu_rows, bmu_cols):
#         # Update the weights using competitive learning for a batch of inputs
#         for i in range(x.size(0)):  # Iterate over the batch
#             for row in range(self.grid_size[0]):
#                 for col in range(self.grid_size[1]):
#                     # Compute the neighborhood function
#                     distance_to_bmu = (row - bmu_rows[i]) ** 2 + (col - bmu_cols[i]) ** 2
#                     neighborhood = torch.exp(-distance_to_bmu / (2 * self.sigma ** 2))

#                     # Update the weights
#                     self.weights[row, col] += self.lr * neighborhood * (x[i] - self.weights[row, col])

#     def train(self, data_in, num_epochs):
#         for epoch in range(num_epochs):
#             # Shuffle the data
#             indices = np.random.permutation(data_in.shape[0])
#             data = data_in[indices]

#             # Train in batches
#             for i in range(0, data.shape[0], self.batch_size):
#                 batch = torch.tensor(data[i:i + self.batch_size], dtype=torch.float32, device=self.device)  # Get a batch of inputs and put on GPU
#                 bmu_rows, bmu_cols = self.find_bmu(batch)  # Find BMUs for the batch
#                 self.update_weights(batch, bmu_rows, bmu_cols)  # Update weights

#                 if i%1 == 0:
#                     sys.stdout.write(f"\rSOM Epoch:{epoch}/{num_epochs-1}, iter_curr:{i}/{data.shape[0]-1}      ") 
#                     sys.stdout.flush() 

#             # Decay learning rate and sigma over time
#             self.lr *= self.lr_epoch_decay
#             self.sigma *= self.sigma_epoch_decay

#     def get_weights(self):
#         return self.weights.cpu().numpy()  # Move weights back to CPU for visualization


