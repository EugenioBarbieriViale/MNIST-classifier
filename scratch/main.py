import numpy as np
from get_data import X_train, Y_train
from network import NN

input_size  = 784
hidden_size = 128
output_size = 10

def normalize(image):
    return image / 255

nn = NN(input_size, hidden_size, output_size)

# for i in range(len(X_train)):
for i in range(1):
    input_layer = normalize(X_train[i])
    label = Y_train[i]

    nn.train(input_layer, label)
