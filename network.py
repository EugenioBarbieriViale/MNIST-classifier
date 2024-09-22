import numpy as np

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.epochs = 1000
        self.learn_rate = 1e-2

        self.in_size  = input_size
        self.hid_size = hidden_size
        self.out_size = output_size

        self.w1 = np.random.rand(self.in_size,  self.hid_size)
        self.w2 = np.random.rand(self.hid_size, self.out_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, v):
        return np.exp(v) / np.exp(v).sum()

    # def softmax(self, x):
    #     exp_element=np.exp(x-x.max())
    #     return exp_element/np.sum(exp_element,axis=0)

    def d_softmax(self, x):
        exp_element=np.exp(x-x.max())
        return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    def forward(self, input_layer):
        self.hidden_net = np.dot(input_layer, self.w1)
        self.hidden_out = self.sigmoid(self.hidden_net)

        self.output_net = np.dot(self.hidden_out, self.w2)
        self.output_out = self.softmax(self.output_net)

    def backward(self, input_layer):
        out_error = 2 * (self.target - self.output_out) * self.d_softmax(self.output_out)
        hid_error = np.dot(self.w2, out_error.T).T * self.d_sigmoid(self.hidden_out)

        w2_delta = np.dot(self.hidden_out.T, out_error)
        w1_delta = np.dot(input_layer.T, hid_error)

        return w2_delta, w1_delta

    def loss(self):
        return ((self.output_out - self.target) ** 2).sum()

    def train(self, input_layer, label):
        self.target = np.zeros((1, 10), np.float32)
        self.target[0][label] = 1

        self.w1 /= 100
        self.w2 /= 100

        for j in range(self.epochs):

            self.forward(input_layer)
            cost = self.loss()

            up_w2, up_w1 = self.backward(input_layer)
            
            self.w2 -= self.learn_rate * up_w2
            self.w1 -= self.learn_rate * up_w1

            print(j, cost)
