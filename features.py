from librosa.core import load, piptrack
from librosa.feature import mfcc, spectral_flatness, rms
import numpy as np
import h5py

def getMFCC(data, rate, win_size, step, file_name):  
    win_size = int(30*rate/1000)
    mfccs = mfcc(y = data, sr = rate, n_mfcc = 12, n_fft = win_size, hop_length = step, n_mels = 40)
    mfccs = mfccs - np.mean(mfccs, axis = 0)
    first_diff = np.diff(mfccs)
    first_diff = np.insert(first_diff, 0, values = [0]*first_diff.shape[0], axis = 1)
    mfccs = np.append(mfccs, first_diff, axis = 0)
    del first_diff
    print(mfccs.shape)
    saveFeatures(mfccs.T, file_name + "_MFCC")
    
def getSpectralFlatness(data, rate, win_size, step, file_name):    
    SFM = spectral_flatness(data, n_fft = win_size, hop_length = step)
    SFM = 10*np.log10(SFM)
    first_diff = np.diff(SFM)
    first_diff = np.insert(first_diff, 0, values = [0], axis = 1)
    SFM = np.append(SFM, first_diff, axis = 0)
    del first_diff
    print(SFM.shape)
    saveFeatures(SFM.T, file_name + "_SFM")
    
def getHarmonicity(data, rate, win_size, step, file_name):
    pitch, mag = piptrack(y = data, sr = rate, n_fft = win_size, hop_length = step)
    del pitch
    harmonicity = []
    for i in mag.T:
        index = np.nonzero(i)
        if(len(index[0])):
            harmonicity.append(np.std(i[index]))
        else:
            harmonicity.append(0)
    del mag
    harmonicity = np.array(harmonicity, ndmin = 2)
    first_diff = np.diff(harmonicity)
    first_diff = np.insert(first_diff, 0, values = [0], axis = 1)
    harmonicity = np.append(harmonicity, first_diff, axis = 0)
    del win_size, first_diff
    print(harmonicity.shape)
    saveFeatures(harmonicity.T, file_name + "_HARMONICITY")
    
def getRMSenergy(data, rate, win_size, step, file_name):
    win_size = int(30*rate/1000)
    rmse = rms(y = data, frame_length = win_size, hop_length = step)
    first_diff = np.diff(rmse)
    first_diff = np.insert(first_diff, 0, values = [0], axis = 1)
    rmse = np.append(rmse, first_diff, axis = 0)
    del first_diff
    print(rmse.shape)
    saveFeatures(rmse.T, file_name + "_RMS")

def editText(file_name):
    label_data = np.loadtxt("amicorpus/AMI_Labels/" + file_name + ".lab", dtype = "str")
    index = np.argwhere(label_data[:, 2] == "sil")
    label_data[index, 2] = 0
    labels = set(label_data[:, 2])
    labels.remove('0')
    for i in labels:
        if(len(i) == 1):
            index = np.argwhere(label_data[:, 2] == i)        
            label_data[index, 2] = 1
        else:
            index = np.argwhere(label_data[:, 2] == i)        
            label_data[index, 2] = 2
    return(label_data.astype(float))

def getLabels(length, a):
    t = 0
    labels = np.zeros(length)
    for j, k in zip(a[:, 1], a[:, 2]):
        if(j != 9999):
            x = int((16*j*1000 - 800)/160)
            labels[t:x] = int(k)
            t = x
        else:
            labels[t:] = int(k)
    return(np.array(labels, ndmin = 2).T)

def saveFeatures(feature, file_name):
    f_name = file_name + ".csv"
    np.savetxt(f_name, feature, delimiter = ",", fmt = "%0.2f")

def getFeatures(files, rate, win_size, step, dset_name):
    features = ['MFCC', 'SFM', 'RMS', 'HARMONICITY']
    dataset = np.array([])
    for folder_name in files:
        path = "amicorpus/" + folder_name + "/audio/" + folder_name + ".Mix-Headset.wav"
        data, rate = load(path, sr = rate)
        getMFCC(data, rate, win_size, step, folder_name)
        getSpectralFlatness(data, rate, win_size, step, folder_name)
        getRMSenergy(data, rate, win_size, step, folder_name)
        getHarmonicity(data, rate, win_size, step, folder_name)
        z = np.array([])
        for i in features:
            path = folder_name + "_" + i + ".csv"
            if(len(z) == 0):
                z = np.array(np.recfromcsv(path, names = None).tolist())
            else:
                z = np.append(z, np.array(np.recfromcsv(path, names = None).tolist()), axis = 1)
        labels = getLabels(z.shape[0], editText(folder_name))
        z = np.append(z, labels, axis = 1)
        saveFeatures(z, folder_name)
        del labels
        if(len(dataset) == 0):
            print("hi")
            dataset = z
        else:
            print("wohoo")
            dataset = np.append(dataset, z, axis = 0)
        del z
    saveFeatures(dataset, "amicorpus_train")
    with h5py.File('amicorpus.hdf5', 'a') as f:
        f.create_dataset(dset_name, data = dataset)

