import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinates
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [0, 1, 2, 3, 4]

# Generate a list of colors for each data point
colors = np.arange(len(x))

# Plot the points with unique colors
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Add a colorbar to show the mapping of colors to values
cbar = plt.colorbar(scatter)
cbar.set_label('Color Index')

# Show the plot
plt.show()
