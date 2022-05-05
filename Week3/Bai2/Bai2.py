import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load data from csv file
data = pd.read_csv('dataset.csv').values
N, d = data.shape           # N - number of samples; d - number of feature
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

# Plot data with scatter
x_accept = x[y[:,0]==1]
x_reject = x[y[:,0]==0]
plt.scatter(x_accept[:, 0], x_accept[:, 1], c='red', edgecolors='none', s=30, label='accept')
plt.scatter(x_reject[:, 0], x_reject[:, 1], c='blue', edgecolors='none', s=30, label='reject')
plt.legend(loc=1)
plt.xlabel('salary (million)')
plt.ylabel('experience (years)')

# Add column of ones to x
x = np.hstack((np.ones((N, 1)), x))

# Initial theta
w = np.array([0.,0.1,0.1]).reshape(-1,1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rate = 0.01

for i in range(1, numOfIteration):
    
	 # Tính giá trị dự đoán
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    # Gradient descent
    w = w - learning_rate * np.dot(x.T, y_predict-y)	 
    print(cost[i])

# Vẽ đường phân cách.
t = 0.5
plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/t-1))/w[2]), 'g')
plt.show()