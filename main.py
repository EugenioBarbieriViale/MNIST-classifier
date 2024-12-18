from keras import *

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# plt.imshow(x_train[0])
# plt.show()

def normalize(data):
  return data / 255.0

def encode_one_hot(labels):
  hot = np.zeros((len(labels), 10))
  for i in range(len(labels)):
    hot[i, labels[i]] = 1.0
  return hot

data = normalize(x_train)
labels = encode_one_hot(y_train)

data_val = normalize(x_test)
labels_val = encode_one_hot(y_test)

def make_model():
  inp = layers.Input(shape=(28,28))
  x = layers.Flatten()(inp)
  x = layers.Dense(128, activation="sigmoid")(x)

  output = layers.Dense(10, activation="softmax")(x)
  return models.Model(inp, output)

model = make_model()
# utils.plot_model(model, show_shapes=True, dpi=70)


# Hyperparameters
epochs = 20
rate = 1e-4
batch_size = 128

loss_history = np.zeros(epochs)
loss_val_history = np.zeros(epochs)
acc_history = np.zeros(epochs)
acc_val_history = np.zeros(epochs)

model.compile(
    optimizer=optimizers.Adam(rate),
    loss=losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

for epoch in range(epochs):
  print("Epoch: ", epoch)
  m = model.fit(data, labels, epochs=1, validation_data=[data_val, labels_val], batch_size=batch_size)

  loss_history[epoch] = m.history['loss'][0]
  loss_val_history[epoch] = m.history['val_loss'][0]
  acc_history[epoch] = m.history['accuracy'][0]
  acc_val_history[epoch] = m.history['val_accuracy'][0]


plt.plot(loss_history, label="loss")
plt.plot(loss_val_history, label="val_loss")
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(acc_history, label="accuracy")
plt.plot(acc_val_history, label="val_accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()


def test(index):
    l = labels[index]
    im = data[index]

    y = model.predict(data_val[index:index+1], verbose=0)[0]
    val = np.max(y)
    for i in range(10):
      guess = i
      if y[i] == val:
        break

    print("Prediction:", guess)
    plt.imshow(data_val[index])
    plt.show()

test(100)
