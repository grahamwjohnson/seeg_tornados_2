import hashlib
import random
import numpy as np
import torch

def hash_to_vector(input_string, num_channels, latent_dim, modifier):
    # Incorporate the modifier into the input string to vary the output
    modified_input = f"{input_string}_{modifier}"

    # Generate a SHA-256 hash from the modified input string
    hash_object = hashlib.sha256(modified_input.encode('utf-8'))
    hash_digest = hash_object.digest()  # 32 bytes (256 bits)

    # If latent_dim > 256, repeat the hash digest to ensure we have enough data
    extended_hash = (hash_digest * ((latent_dim // 32) + 1))[:latent_dim]  # Repeat and slice to exactly latent_dim bytes
    
    # Generate a vector of size latent_dim with values from -1 to 1
    hashed_vector = np.zeros(latent_dim)

    for i in range(latent_dim):
        # Use the i-th byte from the extended hash digest
        byte_value = extended_hash[i]
        
        # Normalize the byte value to the range [-1, 1]
        hashed_vector[i] = (byte_value / 127.5) - 1  # Normalize to [-1, 1]

    # Convert hashed_vector to a PyTorch tensor
    hashed_vector_tensor = torch.tensor(hashed_vector, dtype=torch.float32)

    # Generate a vector of shuffled numbers 0 to num_channels-1
    ordered_vector = list(range(num_channels))
    
    # Set the seed for deterministic shuffling based on the hash of the modified input string
    random.seed(int.from_bytes(hash_digest[:8], 'big'))  # Use first 8 bytes of hash as the seed
    random.shuffle(ordered_vector)  # Shuffle the list in place
    
    return hashed_vector_tensor, ordered_vector


# Example usage
input_string = "example_string"
num_channels = 5
latent_dim = 2048  # Example for large latent_dim
modifier = 42  # This is the additional integer input

hashed_vector_tensor, shuffled_vector = hash_to_vector(input_string, num_channels, latent_dim, modifier)

print("Hashed Vector Tensor:", hashed_vector_tensor)
print("Shuffled Vector:", shuffled_vector)
