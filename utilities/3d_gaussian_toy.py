import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Soft-coded Initialization Variables (from your provided code) ---
num_components = 2
latent_dimension = 3
cube_boundary_min = -5
cube_boundary_max = 5
mean_boundary_min = -3
mean_boundary_max = 3
logvar_min = -2
logvar_max = 3
color_map_name = 'Spectral'
blob_alpha_start = 0.6  # Increased starting alpha for more solid inner shapes
blob_alpha_end = 0.01 # Decreased ending alpha for more transparent outer shapes
num_contours = 4
mean_marker_size = 70
mean_marker_type = 'o'
plot_elevation_angle = 25
plot_azimuthal_angle = -150
background_color = 'white'
figure_size_x = 10
figure_size_y = 8
num_points_surface = 60  # Increased for smoother shading
overlap_factor = 1.0
size_multiplier = 1.7
spread_multiplier = 1
corner_color = 'black'
corner_line_width = 1.0
# --- End of Soft-coded Initialization Variables ---

# --- New Soft-coded Variable for Light Source (Softer Lighting) ---
light_direction = np.array([1, 1, 1])  # More general light direction, closer to viewer
light_direction = light_direction / np.linalg.norm(light_direction) # Normalize
# --- End of New Soft-coded Variable ---

# Define the number of components and latent dimension
K = num_components
latent_dim = latent_dimension

# Define the cube boundaries for visualization
cube_min = cube_boundary_min
cube_max = cube_boundary_max

# Initialize prior means within the inner cube (more spread out)
prior_initial_mean_spread = (mean_boundary_max - mean_boundary_min) / 2
prior_means = ((torch.rand(K, latent_dim) * 2 * prior_initial_mean_spread) + mean_boundary_min) * spread_multiplier

# Vary the prior log variances (for bigger blobs)
prior_logvars = torch.rand(K, latent_dim) * (logvar_max - logvar_min) + logvar_min
prior_stddevs = torch.exp(0.5 * prior_logvars).numpy() * size_multiplier

# Soft and aesthetically pleasing color palette
cmap = cm.get_cmap(color_map_name, K)

fig = plt.figure(figsize=(figure_size_x, figure_size_y))
ax = fig.add_subplot(111, projection='3d')

# Create a LightSource object using azdeg and altdeg
azimuth = np.degrees(np.arctan2(light_direction[1], light_direction[0]))
altitude = np.degrees(np.arcsin(light_direction[2] / np.linalg.norm(light_direction)))
ls = LightSource(azdeg=azimuth, altdeg=altitude)

# Plot each Gaussian component with concentric shapes
for i in range(K):
    mean = prior_means[i].numpy()
    base_stddev = prior_stddevs[i] * overlap_factor
    color = cmap(i / (K - 1) if K > 1 else 0)

    for j in range(1, num_contours + 1):
        # Scale the standard deviation for each concentric shape
        current_stddev = base_stddev * (j / num_contours)

        # Create a grid of points for the surface
        u = np.linspace(0, 2 * np.pi, num_points_surface)
        v = np.linspace(0, np.pi, num_points_surface)
        x = mean[0] + current_stddev[0] * np.outer(np.cos(u), np.sin(v))
        y = mean[1] + current_stddev[1] * np.outer(np.sin(u), np.sin(v))
        z = mean[2] + current_stddev[2] * np.outer(np.ones(num_points_surface), np.cos(v))

        # Calculate alpha based on the contour level
        alpha = blob_alpha_start - (j / (num_contours + 1)) * (blob_alpha_start - blob_alpha_end)

        # Combine the coordinates into vertices
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Calculate surface normals (approximation using cross product of adjacent vectors)
        normals = np.zeros_like(vertices)
        num_u, num_v = x.shape
        for row in range(num_u):
            for col in range(num_v):
                # Get indices of neighboring points (handle boundaries)
                ip = (row + 1) % num_u
                im = (row - 1 + num_u) % num_u
                jp = (col + 1) % num_v
                jm = (col - 1 + num_v) % num_v

                v1 = np.array([x[ip, col] - x[im, col], y[ip, col] - y[im, col], z[ip, col] - z[im, col]])
                v2 = np.array([x[row, jp] - x[row, jm], y[row, jp] - y[row, jm], z[row, jp] - z[row, jm]])

                normal = np.cross(v1, v2)
                normals[row * num_v + col] = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0, 1])

        # Calculate dot product of normals and light direction (diffuse shading)
        intensity = np.dot(normals, light_direction)
        intensity = np.clip(intensity * 0.7 + 0.8, 0.8, 0.9) # Soften and add a bit of ambient
        # Apply shading to the color
        face_colors = cmap(i / (K - 1) if K > 1 else 0)
        shaded_colors = face_colors[:3] * intensity[:, np.newaxis]
        shaded_colors = np.concatenate([shaded_colors, np.full((len(shaded_colors), 1), alpha)], axis=1)

        # Plot the surface as a collection of polygons
        faces = []
        for row in range(num_u - 1):
            for col in range(num_v - 1):
                v1 = row * num_v + col
                v2 = row * num_v + (col + 1)
                v3 = (row + 1) * num_v + (col + 1)
                v4 = (row + 1) * num_v + col
                faces.append([vertices[v1], vertices[v2], vertices[v3], vertices[v4]])

        face_colors_flat = shaded_colors.reshape((num_u, num_v, 4))[:-1, :-1].reshape((-1, 4))
        poly3d = Poly3DCollection(faces, facecolors=face_colors_flat, linewidths=0.1, antialiased=True)
        ax.add_collection3d(poly3d)

    # Plot the mean as a point
    ax.scatter(mean[0], mean[1], mean[2], color='black', s=mean_marker_size, marker=mean_marker_type, alpha=0.8)

# Define cube corners
corners = np.array([
    [cube_min, cube_min, cube_min],
    [cube_max, cube_min, cube_min],
    [cube_min, cube_max, cube_min],
    [cube_max, cube_max, cube_min],
    [cube_min, cube_min, cube_max],
    [cube_max, cube_min, cube_max],
    [cube_min, cube_max, cube_max],
    [cube_max, cube_max, cube_max],
])

# Plot lines connecting the cube corners
lines = [
    [0, 1], [0, 2], [0, 4],
    [1, 3], [1, 5],
    [2, 3], [2, 6],
    [3, 7],
    [4, 5], [4, 6],
    [5, 7],
    [6, 7],
]

for line in lines:
    p1 = corners[line[0]]
    p2 = corners[line[1]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=corner_color, linewidth=corner_line_width)

# Set the cube boundaries for the plot
ax.set_xlim(cube_boundary_min, cube_boundary_max)
ax.set_ylim(cube_boundary_min, cube_boundary_max)
ax.set_zlim(cube_boundary_min, cube_boundary_max)

# Enforce equal aspect ratio for the axes
ax.set_aspect('equal')

# Make the background all white
ax.set_facecolor(background_color)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Turn off axis ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_title(f'{K} Gaussian Prior Components (Spectral Colormap) - Softer Shading')
ax.view_init(elev=plot_elevation_angle, azim=plot_azimuthal_angle)
plt.tight_layout()
plt.show()