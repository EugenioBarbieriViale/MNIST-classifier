import numpy as np

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.epochs = 1
        self.learn_rate = 1e-3

        self.in_size  = input_size
        self.hid_size = hidden_size
        self.out_size = output_size

        self.w1 = np.random.rand(self.in_size,  self.hid_size)
        self.w2 = np.random.rand(self.hid_size, self.out_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        # return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softamx(self, x):
        return np.exp(x) / np.exp(x).sum()

#     def d_softamx(self, x):

    def forward(self, input_layer):
        self.hidden_net = np.dot(input_layer, self.w1)
        self.hidden_out = self.sigmoid(self.hidden_net)

        self.output_net = np.dot(self.hidden_out, self.w2)
        self.output_out = self.sigmoid(self.output_net)

    def backward(self, input_layer):
        # gradient of output layer with respect to loss function

        out_error = 2 * (self.output_out - self.target) * self.d_sigmoid(self.output_out) # maybe _net
        update_w2 = np.dot(self.hidden_out.T, out_error)

        hid_error = (np.dot(self.w2, out_error.T)).T * self.d_sigmoid(self.hidden_out) # maybe _net
        update_w1 = np.dot(input_layer.T, hid_error)

        return update_w2, update_w1

    def loss(self):
        return ((self.output_out - self.target) ** 2).sum()

    def train(self, input_layer, label):
        self.target = np.zeros((1, 10), np.float32)
        self.target[0][label] = 1

        for j in range(self.epochs):

            self.forward(input_layer)
            cost = self.loss()

            up_w2, up_w1 = self.backward(input_layer)

            print(up_w2)
            print(up_w1)
            # print(cost)

