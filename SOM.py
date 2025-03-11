import torch
import torch.nn as nn
import numpy as np

class SOM(nn.Module):
    def __init__(self, grid_size, input_dim, lr=0.5, sigma=1.0, device='cuda'):
        super(SOM, self).__init__()
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr = lr
        self.sigma = sigma
        self.device = device

        # Initialize the weights (neurons) randomly and move to GPU
        self.weights = torch.randn(grid_size[0], grid_size[1], input_dim, device=self.device)

    def forward(self, x):
        # Compute the squared Euclidean distance between input and weights
        distances = torch.sum((self.weights - x) ** 2, dim=2)
        return distances

    def find_bmu(self, x):
        # Find the Best Matching Unit (BMU) for a given input
        distances = self.forward(x)
        bmu_indices = torch.argmin(distances, dim=1)
        return bmu_indices

    def update_weights(self, x, bmu_indices):
        # Update the weights using competitive learning
        for i, bmu_idx in enumerate(bmu_indices):
            bmu_row, bmu_col = bmu_idx // self.grid_size[1], bmu_idx % self.grid_size[1]
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    # Compute the neighborhood function
                    distance_to_bmu = (row - bmu_row) ** 2 + (col - bmu_col) ** 2
                    neighborhood = torch.exp(-distance_to_bmu / (2 * self.sigma ** 2))

                    # Update the weights
                    self.weights[row, col] += self.lr * neighborhood * (x[i] - self.weights[row, col])

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            for x in data:
                x = torch.tensor(x, dtype=torch.float32, device=self.device)  # Move input to GPU
                bmu_indices = self.find_bmu(x)
                self.update_weights(x, bmu_indices)

            # Decay learning rate and sigma over time
            self.lr *= 0.99
            self.sigma *= 0.99

            if epoch%10 == 0:
                sys.stdout.write(f"\rSOM Epoch: {epoch}/{num_epochs}       ") 
                sys.stdout.flush() 

    def get_weights(self):
        return self.weights.cpu().numpy()  # Move weights back to CPU for visualization