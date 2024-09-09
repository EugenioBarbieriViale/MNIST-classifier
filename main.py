import numpy as np
from get_data import X_train, Y_train

input_size = 784
hidden_size = 128
output_size = 10

epochs = 1
learn_rate = 1e-2
h = 1e-2

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def d_sigmoid(x):
    return np.exp(-x)/(np.exp(-x)**2 + 1)

def forward(w1, w2):
    hidden_net = np.dot(input_layer, w1)
    hidden_out = sigmoid(hidden_net)

    output_net = np.dot(hidden_out, w2)
    output_out = sigmoid(output_net)

    cost = ((output_out - target) ** 2).sum()
    return cost

w1 = np.random.rand(784, 128)
w2 = np.random.rand(128,  10)

g1 = np.zeros((784, 128))
g2 = np.zeros((128,  10))

# for i in range(len(X_train)):
for i in range(1):
    label = Y_train[i]
    input_layer = X_train[i]

    target = [1 if n == (label-1) else 0 for n in range(output_size)]

    for j in range(epochs):
        cost = forward(w1, w2)
        print(cost)
