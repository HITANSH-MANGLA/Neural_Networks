import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Load dataset
text = np.loadtxt("data_vector_quantization.txt")
data = text[:, 0:2]
labels = text[:, 2:]

# Config
# Directly define how many neurons for each class (must sum to total neurons)
class_counts = [3, 3, 2, 2]  # total = 10 neurons
num_input_neurons = sum(class_counts)

# Convert to proportions (only for API compliance — won't break slicing)
proportions = [c / num_input_neurons for c in class_counts]

# Create LVQ network
nn = nl.net.newlvq(nl.tool.minmax(data), num_input_neurons, proportions)

# Train the network
_ = nn.train(data, labels, epochs=500, goal=-1)

# Create grid for visualization
xx, yy = np.meshgrid(np.arange(0, 10, 0.2), np.arange(0, 10, 0.2))
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
grid_xy = np.concatenate((xx, yy), axis=1)

# Predict on grid
grid_eval = nn.sim(grid_xy)

# Split input data by class
class_1 = data[labels[:, 0] == 1]
class_2 = data[labels[:, 1] == 1]
class_3 = data[labels[:, 2] == 1]
class_4 = data[labels[:, 3] == 1]

# Split grid prediction by class
grid_1 = grid_xy[grid_eval[:, 0] == 1]
grid_2 = grid_xy[grid_eval[:, 1] == 1]
grid_3 = grid_xy[grid_eval[:, 2] == 1]
grid_4 = grid_xy[grid_eval[:, 3] == 1]

# Plot input points
plt.plot(class_1[:, 0], class_1[:, 1], 'ko',
         class_2[:, 0], class_2[:, 1], 'ko',
         class_3[:, 0], class_3[:, 1], 'ko',
         class_4[:, 0], class_4[:, 1], 'ko')

# Plot grid predictions
plt.plot(grid_1[:, 0], grid_1[:, 1], 'm.',
         grid_2[:, 0], grid_2[:, 1], 'bx',
         grid_3[:, 0], grid_3[:, 1], 'c.',
         grid_4[:, 0], grid_4[:, 1], 'y+')

plt.axis([0, 10, 0, 10])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Vector Quantization (LVQ)')
plt.grid(True)
plt.show()
