import numpy as np
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from init import pad_range, time_range, device
from data import preprocessing
preprocessing._VERSION = 'data_v4'


def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:, 2:4] % 1
    features = (features[:, :3] - np.array([[0.0, 0.0, 162.5]])) / np.array([[20.0, 60.0, 127.5]])
    return np.concatenate([features, bin_fractions], axis=-1)


def get_data():
    data, features = preprocessing.read_csv_2d(
        filename='data/digits.csv', pad_range=pad_range, time_range=time_range
    )

    data_scaled = np.log10(1 + data).astype('float32')
    features_processed = preprocess_features(features).astype('float32')
    return train_test_split(data_scaled, features_processed, test_size=0.25, random_state=42)
    # data_train = Variable(torch.from_numpy(data_train), requires_grad=True).to(device)
    # data_test = Variable(torch.from_numpy(data_test), requires_grad=True).to(device)
    # features_train = Variable(torch.from_numpy(features_train), requires_grad=True).to(device)
    # features_test = Variable(torch.from_numpy(features_test), requires_grad=True).to(device)
    # return data_train, data_test, features_train, features_test
