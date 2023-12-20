import matplotlib.pyplot as plt

# Create a figure with a 1x4 subplot grid
fig = plt.figure(figsize=(50,50))

# Add a subplot at position 1
ax1 = fig.add_subplot(141)
ax1.plot([1, 2, 3], [4, 5, 6])

# Add a subplot at position 2
ax2 = fig.add_subplot(142)
ax2.plot([1, 2, 3], [4, 5, 6])

# Add a subplot at position 3
ax3 = fig.add_subplot(143)
ax3.plot([1, 2, 3], [4, 5, 6])

# Add a subplot at position 4
ax4 = fig.add_subplot(144)
ax4.plot([1, 2, 3], [4, 5, 6])

# Show the plot
plt.show()
