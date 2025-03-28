import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F



# ######### POSTERIOR ########

# def gumbel_softmax_sample(logits, temperature):
#     """Generates a sample from a Gumbel-Softmax distribution."""
#     noise = torch.rand_like(logits).log() - torch.rand_like(logits).log()
#     return F.softmax((logits + noise) / temperature, dim=-1)

# def negative_entropy_loss_example(mogpreds, weight):
#     """
#     Toy example of negative entropy loss based on your function.
#     mogpreds: MoG component probabilities, shape (batch_size, T, K)
#     weight: Weight for the entropy loss
#     """
#     # Ensure mogpreds is a valid probability distribution (for the example)
#     assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
#     assert torch.allclose(mogpreds.sum(dim=-1, keepdim=True), torch.ones_like(mogpreds.sum(dim=-1, keepdim=True))), "mogpreds does not sum to 1"

#     # Clamp mogpreds
#     mogpreds = torch.clamp(mogpreds, min=1e-10, max=1.0)

#     # Compute the average probability of each component across all samples (simplified for toy example)
#     aggregated_probs = mogpreds.mean(dim=(0, 1))

#     # Compute entropy
#     entropy = -torch.mean(aggregated_probs * torch.log(aggregated_probs))

#     return -weight * entropy

# def cosine_diversity_loss_example(mogpreds, weight):
#     """
#     Toy example of cosine diversity loss based on your function.
#     mogpreds: MoG component probabilities, shape (batch_size, T, K)
#     weight: Weight for the diversity loss
#     """
#     # Ensure mogpreds is a valid probability distribution (for the example)
#     assert torch.all(mogpreds >= 0), "mogpreds contains negative values"
#     assert torch.allclose(mogpreds.sum(dim=-1), torch.ones_like(mogpreds.sum(dim=-1))), "mogpreds does not sum to 1"

#     # Clamp mogpreds
#     mogpreds = torch.clamp(mogpreds, min=1e-10, max=1.0)

#     # Compute the mean prediction for each sequence (simplified for toy example with batch_size=2)
#     mean_mogpreds = mogpreds.mean(dim=1)

#     # For a toy example with batch_size=2, we can directly compute cosine similarity
#     if mean_mogpreds.shape[0] >= 2:
#         vec1 = mean_mogpreds[0]
#         vec2 = mean_mogpreds[1]
#         cosine_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
#         diversity_loss = cosine_sim
#     else:
#         diversity_loss = torch.tensor(0.0)  # No diversity if only one sequence

#     return weight * diversity_loss

# # Set the number of components (K)
# n_components = 8
# latent_dim = 1024
# batch_size = 8  # For the diversity loss example
# time_steps = 512  # For the mogpreds shape

# # Create dummy logits with the correct shape [batch_size, time_steps, n_components]
# logits = torch.randn(batch_size, time_steps, n_components) * 2

# # Define a range of temperatures to visualize
# temperatures = np.linspace(2.0, 0.5, 200)

# # Store the output distributions and losses for each temperature
# distributions = []
# entropy_losses = []
# diversity_losses = []

# # Define fixed weights for the loss terms (for this visualization)
# entropy_weight = 0.5
# diversity_weight = 0.1

# # Generate samples and calculate losses for each temperature
# for temp in temperatures:
#     # Sample mogpreds using Gumbel-Softmax
#     mogpreds = gumbel_softmax_sample(logits, temp)
#     distributions.append(mogpreds.detach().numpy())

#     # Calculate negative entropy loss
#     entropy_loss = negative_entropy_loss_example(mogpreds, entropy_weight)
#     entropy_losses.append(entropy_loss.item())

#     # Calculate cosine diversity loss
#     diversity_loss = cosine_diversity_loss_example(mogpreds, diversity_weight)
#     diversity_losses.append(diversity_loss.item())

# # Convert the list of numpy arrays to a single array
# distributions_array = np.array(distributions)

# # Create the plot
# fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# # Plot the probability (weight) of each component across different temperatures (for the first batch and time step)
# axs[0].set_title('Effect of Temperature on Gumbel-Softmax Component Weights (Batch 0, Time 0)')
# for i in range(n_components):
#     axs[0].plot(temperatures, distributions_array[:, 0, 0, i], label=f'Component {i+1}')
# axs[0].set_ylabel('Weight (Probability)')
# axs[0].legend(title='Components', fontsize='small')
# axs[0].grid(True)
# axs[0].invert_xaxis()

# # Plot the Negative Entropy Loss
# axs[1].plot(temperatures, entropy_losses, label=f'Weight: {entropy_weight}')
# axs[1].set_ylabel('Negative Entropy Loss')
# axs[1].grid(True)
# axs[1].legend()
# axs[1].invert_xaxis()

# # Plot the Cosine Diversity Loss
# axs[2].plot(temperatures, diversity_losses, label=f'Weight: {diversity_weight}')
# axs[2].set_xlabel('Temperature (τ)')
# axs[2].set_ylabel('Cosine Diversity Loss')
# axs[2].grid(True)
# axs[2].legend()
# axs[2].invert_xaxis()

# plt.tight_layout()
# plt.show()




######### PRIOR ########

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def gumbel_softmax_sample(logits, temperature):
    """Generates a sample from a Gumbel-Softmax distribution."""
    noise = torch.rand_like(logits).log() - torch.rand_like(logits).log()
    return F.softmax((logits + noise) / temperature, dim=-1)

# Set the number of components
n_components = 8  # You can change this value

# Define a range of temperatures to visualize (from 2.0 to 0.5)
temperatures = np.linspace(2.0, 0.5, 100)

# --- Scenario 1: Uniform weightlogits ---
uniform_weightlogits = torch.ones(n_components) / n_components
uniform_logits_for_gumbel = torch.log_softmax(torch.tensor(uniform_weightlogits), dim=0)
uniform_distributions = []
for temp in temperatures:
    sample = gumbel_softmax_sample(uniform_logits_for_gumbel.unsqueeze(0), temp).squeeze(0)
    uniform_distributions.append(sample.numpy())
uniform_distributions_array = np.array(uniform_distributions)

# --- Scenario 2: Non-uniform weightlogits (always random) ---
non_uniform_weightlogits = torch.rand(n_components)

# Ensure the non-uniform weights sum to approximately 1
non_uniform_weightlogits = non_uniform_weightlogits / non_uniform_weightlogits.sum()

non_uniform_logits_for_gumbel = torch.log_softmax(non_uniform_weightlogits, dim=0)
non_uniform_distributions = []
for temp in temperatures:
    sample = gumbel_softmax_sample(non_uniform_logits_for_gumbel.unsqueeze(0), temp).squeeze(0)
    non_uniform_distributions.append(sample.numpy())
non_uniform_distributions_array = np.array(non_uniform_distributions)

# Create the plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot for Uniform weightlogits
axs[0].set_title('Gumbel-Softmax Output with Uniform Weightlogits')
for i in range(n_components):
    axs[0].plot(temperatures, uniform_distributions_array[:, i], label=f'Component {i+1}')
axs[0].set_ylabel('Weight (Probability)')
axs[0].legend(title='Components', fontsize='small')
axs[0].grid(True)
axs[0].set_xlim(2.0, 0.5)

# Plot for Non-uniform weightlogits
axs[1].set_title('Gumbel-Softmax Output with Random Non-Uniform Weightlogits')
for i in range(n_components):
    axs[1].plot(temperatures, non_uniform_distributions_array[:, i], label=f'Component {i+1}')
axs[1].set_xlabel('Temperature (τ)')
axs[1].set_ylabel('Weight (Probability)')
axs[1].legend(title='Components', fontsize='small')
axs[1].grid(True)
axs[1].set_xlim(2.0, 0.5)

plt.tight_layout()
plt.show()