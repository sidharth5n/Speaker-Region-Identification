import random as rnd
import pickle
import numpy as np

def softmax(z):
    d = np.max(z, axis = 0)
    num = np.exp(z - d)
    den = np.sum(num, axis = 0)
    return num / den

def ReLU(z):
    return np.maximum(z, 0)

def cross_entropy(y, y_hat):
    loss = -np.mean(np.log(y_hat[y.astype(int), np.arange(y_hat.shape[1])]))
    return loss

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

def data_generator(training_data, batch_size, shuffle = True):
    n, m = training_data.shape
    indices = [*range(n)]
    if shuffle:
        rnd.shuffle(indices)
    index = 0
    remaining = n
    while remaining > 0:
        size = min(batch_size, remaining)
        buffer = np.zeros((m, size))
        for i in range(size):
            if index >= n:
                index = 0
                if shuffle:
                    rnd.shuffle(indices)
            buffer[:, i] = training_data[indices[index]]
            index += 1
        remaining -= size
        yield buffer

def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
