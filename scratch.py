import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create a simple 1D waveform (e.g., sine wave)
x = np.linspace(0, 2 * np.pi, 50)  # 50 samples
y = np.sin(x)  # Sine wave

# Convert to torch tensor
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 50)

# Set up nn.Upsample with 'linear' mode to double the size
upsample = nn.Upsample(scale_factor=2, mode='nearest')

# Apply upsampling
upsampled_tensor = upsample(y_tensor)

# Convert back to numpy for plotting
upsampled_waveform = upsampled_tensor.squeeze().detach().numpy()

# Create a new x for the upsampled waveform (double the length)
x_upsampled = np.linspace(0, 2 * np.pi, len(upsampled_waveform))

# Plot original and upsampled waveforms
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Waveform (50 samples)', linestyle='--')
plt.plot(x_upsampled, upsampled_waveform, label='Upsampled Waveform (100 samples)', linestyle='-')
plt.legend()
plt.title("1D Waveform Upsampling using nn.Upsample with 'linear' interpolation")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
