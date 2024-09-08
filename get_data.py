# Download mnist dataset here: 
# https://github.com/cvdfoundation/mnist


import gzip
import numpy as np

path = "data/"

train_image_file = path + "train-images-idx3-ubyte.gz"
train_label_file = path + "train-labels-idx1-ubyte.gz"

test_image_file  = path + "t10k-images-idx3-ubyte.gz"
test_label_file  = path + "t10k-labels-idx1-ubyte.gz"

def process(name_file):
    with open(name_file, "rb") as f:
        data = f.read()
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

# X_train = process(train_image_file)[0x10:].reshape((-1, 28, 28))
X_train = process(train_image_file)[0x10:].reshape((-1, 1, 28*28))
Y_train = process(train_label_file)[8:]

# X_test = process(test_image_file)[0x10:].reshape((-1, 28, 28))
X_test = process(test_image_file)[0x10:].reshape((-1, 1, 28*28))
Y_test = process(test_label_file)[8:]
