import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

text = np.loadtxt(r"C:\Users\Lenovo\Desktop\python using ai\data_perceptron.txt")

data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

plt.figure()
plt.scatter(data[:,0], data[:, 1])
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.title('Input data')

dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1
num_output = labels.shape[1]

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)
error_progress = perceptron.train(data, labels, epochs=100, show=20,lr=0.03)
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number_Epochs')
plt.ylabel('Training_Error')
plt.title('Training Error Progression')
plt.grid()
plt.show()
