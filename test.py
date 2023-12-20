import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinates
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [0, 1, 2, 3, 4]

# Plot the points
ax.scatter(x, y, z, c='r', marker='o')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()
