import features
import h5py
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.preprocessing import StandardScaler
import models

train = ["ES2010a", "IB4011", "ES2005d", "IS1004a", "EN2006b"]
test = ["EN2003a", "IS1002c"]
rate = 16000
step = int(10*rate/1000)
win_size = int(30*rate/1000)

# Extracting features from audio file
for x, y in zip([train, test], ['train', 'test']):
    features.getFeatures(x, rate, win_size, step, y)
del train, test, rate, step, win_size

# Balancing the training dataset
with h5py.File('amicorpus.hdf5', 'r+') as f:
    data = f['train']
    sm = SMOTEENN()
    data, label = sm.fit_sample(data[:, :-1], data[:, -1])
    data = np.append(data, label, axis = 1)
    f.create_dataset('train_balanced', data = data)
    del data

# GMM with 64 components for each class
with h5py.File("amicorpus.hdf5", 'r+') as f:
    train_data = f['train_balanced']
    data_train = np.array([])
    test_data = f['test']
    data_test = np.array([])
    for i in range(3):
        index = np.argwhere(train_data[:, -1] == i).T[0]
        g = GaussianMixture(64, covariance_type = 'full')
        g.fit(train_data[index,:-1])
        del index
        if(len(data_train) == 0):
            data_train = g.predict_proba(train_data[:, :-1])
            data_test = g.predict_proba(test_data[:, :-1])
        else:
            data_train = np.append(data_train, g.predict_proba(train_data[:, :-1]), axis = 1)
            data_test = np.append(data_test, g.predict_proba(test_data[:, :-1]), axis = 1)
        joblib.dump(g, 'g' + str(i))
    data_train = np.append(data_train, np.array(train_data[:, -1], ndmin = 2).T, axis = 1)
    f.create_dataset('Gtrain', data = data_train)
    del data_train
    data_test = np.append(data_test, np.array(test_data[:, -1], ndmin = 2).T, axis = 1)
    f.create_dataset('Gtest', data = data_test)
    del data_test

# Scaling the data to zero mean, unit variance
with h5py.File("amicorpus.hdf5", 'r+') as f:
    train_data_gmm = f['Gtrain']
    test_data_gmm = f['Gtest']
    scale = StandardScaler()
    scale.fit(train_data_gmm[:, :-1])
    train_data = scale.transform(train_data_gmm[:, :-1])
    train_data = np.append(train_data, np.array(train_data_gmm[:, -1], ndmin = 2).T, axis = 1)
    dset = f.create_dataset("Gtrain_scaled", data = train_data)
    del train_data
    test_data = scale.transform(test_data_gmm[:, :-1])
    test_data = np.append(test_data, np.array(test_data_gmm[:, -1], ndmin = 2).T, axis = 1)
    dset = f.create_dataset("Gtest_scaled", data = test_data.T)
    del test_data

# MLP with 1 hidden layer
with h5py.File("amicorpus.hdf5", 'r') as f:
    train_data = f['Gtrain_scaled']
    test_data = f['Gtest_scaled']
    net = models.Network([train_data.shape[1] - 1, 400, 3])
    losses = net.SGD(train_data, test_data)
