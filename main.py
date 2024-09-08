import numpy as np
from get_data import X_train, Y_train

input_size = 784
hidden_size = 128
output_size = 10

epochs = 1
learn_rate = 1e-2

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def d_sigmoid(x):
    return np.exp(-x)/(np.exp(-x)**2 + 1)


def init_weights(rows, cols):
    return np.random.rand(rows, cols)

w1 = init_weights(784,128)
w2 = init_weights(128,10)

for i in range(epochs):
    label = Y_train[i]
    input_layer = X_train[i]

    hidden_layer = sigmoid(np.dot(input_layer, w1))
    output_layer = sigmoid(np.dot(hidden_layer, w2))

    target = [1 if (n+1) == label else 0 for n in range(output_size)]
    loss = (output_layer - target) * d_sigmoid(output_layer)
    cost = loss.sum()

