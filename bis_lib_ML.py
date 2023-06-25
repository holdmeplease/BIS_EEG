import numpy as np

def feature_norm(train_x, test_x):
    tmp_x = np.concatenate((train_x, test_x), axis=0)
    for i in range(tmp_x.shape[1]):
        tmp_x[:,i] = (tmp_x[:,i] - np.mean(tmp_x[:,i])) / np.std(tmp_x[:,i])
    train_x_norm = tmp_x[0:train_x.shape[0],:]
    test_x_norm = tmp_x[train_x.shape[0]:,:]
    return train_x_norm, test_x_norm