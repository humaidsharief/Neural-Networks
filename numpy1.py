import numpy as np

inputs = [5, 2, 3]
weights = [0.6, 0.5, -0.2]
bias = -2.75

weighted_input = [inputs[i] * weights[i] for i in range (len(inputs))]
total_input = sum(weighted_input) + bias

def sigmoid (x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

activated_output = tanh(total_input)
#print("Astronaut Mood Level:", activated_output)