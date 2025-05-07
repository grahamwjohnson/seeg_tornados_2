import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def involuted_clamped_bump(x, y, A=0.0, w=2, S=1.0):
    r = np.sqrt(x**2 + y**2)
    r_scaled = r / S  # Scale the radial distance to make the bump bigger
    R = 2 * A       # ring radius
    center = A * np.exp(-r_scaled**2)                          # center depth increases with A
    ring = np.exp(-((r_scaled - R)**2) / (w**2))               # outer ring
    z = ring - center
    return np.maximum(z, 0)                             # clamp to non-negative

# Create meshgrid
size = 2048
lim = 32
S_max = lim/2
x = np.linspace(-lim, lim, size)
y = np.linspace(-lim, lim, size)
X, Y = np.meshgrid(x, y)

# Set up the 3x3 grid for A and S combinations
A_values = [0.0, 0.5, 1.0]  # Values of A
S_values = [S_max/4, S_max/2, S_max]  # Values of S

fig = plt.figure(figsize=(15, 10))

# Loop over A and S values to create the 3x3 grid of plots
for i, A in enumerate(A_values):
    for j, S in enumerate(S_values):
        ax = fig.add_subplot(3, 3, i * 3 + j + 1, projection='3d')
        Z = involuted_clamped_bump(X, Y, A, S=S)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', antialiased=True)
        
        # Set titles and labels
        ax.set_title(f"A = {A}, S = {S}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Amplitude')
        
        # Optional: Set axis limits
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([0, np.max(Z)])
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

plt.tight_layout()
plt.show()
