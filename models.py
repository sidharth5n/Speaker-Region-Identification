import os
import numpy as np
from utils import *
from sklearn.metrics import f1_score

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.load_weights()

    def SGD(self, train_dset, test_dset = None, batch_size = 32, epochs = 10, learning_rate = 0.01):
        losses = list()
        for epoch in range(epochs):
            loss = 0
            for data in data_generator(train_dset, batch_size):
                x, y = data[:-1], data[-1]
                outputs = self.forward(x)
                delta_b, delta_w = self.backward(x, y, outputs)
                self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, delta_w)]
                self.biases = [b - learning_rate * db for b, db in zip(self.biases, delta_b)]
                loss += cross_entropy(y, outputs[-1])
            print("Epoch {}/{} : Loss = {:0.3f}".format(epoch + 1, epochs, loss))
            save('weights_epoch_' + str(epoch + 1), self.biases)
            save('biases_epoch_' + str(epoch + 1), self.biases)
            losses.append(loss)

        if test_dset is not None:
            loss, f1_score, accuracy = self.evaluate(data_generator(test_dset))
            print("Test : Loss = {:0.3f}, Accuracy = {:0.3f}, F1 Scores = {:0.3f}, {:0.3f}, {:0.3f}".format(loss, accuracy, f1_score[0], f1_score[1], f1_score[2]))

        return losses

    def forward(self, x):
        outputs = [x]
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            outputs.append(ReLU(np.dot(w, outputs[-1])+b))
        outputs.append(softmax(np.dot(self.weights[-1], outputs[-1]) + self.biases[-1]))
        return outputs

    def backward(self, x, y, outputs):
        n = x.shape[1]

	# one hot vector of the target labels
        u = np.zeros((self.sizes[-1], n))
        u[y.astype(int), np.arange(u.shape[1])] = 1
        y = u

        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        dZ = outputs[-1] - y
        del_w[-1] = np.dot(dZ, outputs[-2].T) / n
        del_b[-1] = np.sum(dZ, axis = 1, keepdims = True) / n
        temp = np.dot(self.weights[-1].T, dZ)

        for i in range(self.num_layers - 2, 1, -1):
            dZ = temp * (outputs[i - 1] > 0).astype(float)
            del_w[i - 1] = np.dot(dZ, x.T) / n
            del_b[i - 1] = np.sum(dZ, axis = 1, keepdims = True) / n
            if i > 1:
                temp = np.dot(self.weights[i - 2].T, dZ)

        return (del_b, del_w)

    def evaluate(self, data_loader):
        y, y_hats = list(), list()
        loss = 0
        for x in data_loader:
            y_hat = self.forward(x[:-1])[-1]
            loss += cross_entropy(x[:-1], y_hat) * y_hat.shape[1]
            y = np.append(y, x[:-1])
            y_hats = np.append(y_hats, np.argmax(y_hat, axis = 0))
        f1 = f1_score(y, y_hat, average = None)
        acc = accuracy(y, y_hat)
        loss /= len(y)
        return loss, f1, acc

    def load_weights(self):
        files = os.listdir(os.getcwd())
        weights_file, biases_file = list(), list()
        for file in files:
            if file.startswith('weights_epoch_'):
                weights_file.append(file)
            elif file.startswith('biases_epoch_'):
                biases_file.append(file)
        if weights_file and biases_file:
            weights_file.sort()
            biases_file.sort()
            self.weights = load(weights_file[-1])
            self.biases = load(biases_file[-1])
