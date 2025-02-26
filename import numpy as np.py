import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

np.random.seed(42)  
w1 = np.random.uniform(-0.5, 0.5, (2, 2))
w2 = np.random.uniform(-0.5, 0.5, (2, 1))  

b1 = 0.5  
b2 = 0.7  

inputs = np.array([[0.05, 0.1]])

net_h1 = np.dot(inputs, w1) + b1
out_h1 = tanh(net_h1)

net_o1 = np.dot(out_h1, w2) + b2
output = tanh(net_o1)

print("The answer is:", output)