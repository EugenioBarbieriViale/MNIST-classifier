import numpy as np

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.epochs = 100
        self.learn_rate = 1e-3

        self.in_size  = input_size
        self.hid_size = hidden_size
        self.out_size = output_size

        self.w1 = np.random.rand(self.in_size,  self.hid_size)
        self.w2 = np.random.rand(self.hid_size, self.out_size)

    def sigmoid(self, x):
        return (1/(1 + np.exp(-x)))

    def d_sigmoid(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def forward(self, input_layer):
        hidden_net = np.dot(input_layer, self.w1)
        hidden_out = self.sigmoid(hidden_net)

        output_net = np.dot(hidden_out, self.w2)
        output_out = self.sigmoid(output_net)

        return output_out

    def loss(self, output, label):
        self.target = [1.0 if n == (label-1) else 0.0 for n in range(self.out_size)]
        return ((output - self.target) ** 2).sum()
    
    def train(self, input_layer, label):
        for j in range(self.epochs):

            out = self.forward(input_layer)
            cost = self.loss(out, label)

            print(cost)
