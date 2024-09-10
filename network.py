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

    def back_forward(self, input_layer, target):
        hidden_net = np.dot(input_layer, self.w1)
        hidden_out = self.sigmoid(hidden_net)

        output_net = np.dot(hidden_out, self.w2)
        output_out = self.sigmoid(output_net)

        error = 2 * (output_out - target) / output_out.shape[0] * self.d_sigmoid(output_net)
        update_l2 = hidden_out.T @ error

        error = ((self.w2).dot(error.T)).T * self.d_sigmoid(hidden_net)
        update_l1 = input_layer.T @ error

        print(hidden_net)

        return output_out, update_l1, update_l2
    
    def train(self, input_layer, label):
        target = [1.0 if n == (label-1) else 0.0 for n in range(self.out_size)]

        for j in range(self.epochs):
            out, u1, u2 = self.back_forward(input_layer, target)

            self.w1 -= self.learn_rate * u1
            self.w2 -= self.learn_rate * u2
        
            cost = ((out - target) ** 2).sum()
            print(cost)
