import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load data from csv file
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

# Plot data with scatter
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='accept')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='reject')
plt.legend(loc=1)
plt.xlabel('salary (million)')
plt.ylabel('experience (year)')

# Add 1 column into  x
x = np.hstack((np.ones((N, 1)), x))

# Init theta
w = np.array([0.,0.1,0.1]).reshape(-1,1)

# Number of Iteration
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rate = 0.01

for i in range(1, numOfIteration):   
	 # Calculate predict of y
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))

    # Gradient descent
    w = w - learning_rate * np.dot(x.T, y_predict-y)	 
    print(cost[i])

# Draw Sector
t = 0.5
plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/t-1))/w[2]), 'g')
plt.show()