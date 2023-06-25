import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.signal import blackman, hamming, detrend

import antropy as ant
import mne_connectivity

from bis_lib_noise import noise_handle, noise_handle_svd

# Permutation entropy
def get_pe(eeg_data):
    pe = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            pe[i,j] = ant.perm_entropy(eeg_data[i,j,:], normalize=True)
    print('pe.shape', pe.shape)
    for i in range(pe.shape[1]):
        plt.plot(pe[:,i])
    plt.show()
    return pe

# Approximate entropy
def get_ae(eeg_data):
    ae = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            ae[i,j] = ant.app_entropy(eeg_data[i,j,:])
    print('ae.shape', ae.shape)
    for i in range(ae.shape[1]):
        plt.plot(ae[:,i])
    plt.show()
    return ae

# Sample entropy
def get_se(eeg_data):
    se = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            se[i,j] = ant.sample_entropy(eeg_data[i,j,:])
    print('se.shape', se.shape)
    for i in range(se.shape[1]):
        plt.plot(se[:,i])
    plt.show()
    return se

# Lempel-Ziv complexity
def get_lzc(eeg_data):
    lzc = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            x_bin = eeg_data[i,j,:].copy()*0
            thr = np.mean(eeg_data[i,j,:])
            x_bin[np.where(eeg_data[i,j,:] >= thr)] = 1
            lzc[i,j] = ant.lziv_complexity(x_bin, normalize=True)
    print('lzc.shape', lzc.shape)
    for i in range(lzc.shape[1]):
        plt.plot(lzc[:,i])
    plt.show()
    return lzc

# connectivity
def get_conn_svd(eeg_data, fs):
    freqs_num = 4
    f_start = [1,4,8,14]
    f_end = [4,8,14,30]
    svd = np.zeros(((freqs_num, eeg_data.shape[0])))
    for i in range(eeg_data.shape[0]):
        eeg_tmp = eeg_data[i,:,:][np.newaxis, :, :]
        for j in range(freqs_num):
            conn = mne_connectivity.spectral_connectivity_time(eeg_tmp, [f_start[j], f_end[j]], method='coh', 
                                                               sfreq=fs, faverage=True)
            print('conn.shape', conn.shape)
            conn = np.squeeze(conn.get_data()[0])
            conn_matrix = conn.reshape((eeg_data.shape[1], -1))
            conn_matrix[np.triu_indices(eeg_data.shape[1], 1)] = conn_matrix[np.tril_indices(eeg_data.shape[1], -1)]
            conn_matrix[np.diag_indices(eeg_data.shape[1])] = 1
            s, _ = np.linalg.eig(conn_matrix)
            svd[j,i] = np.abs(np.max(s))
    return svd

# 60s 进行一次计算
# 最好不要直接替换EEG，因为可能会人为加入高频噪声
# 理想的策略是1s计算一次，每60s取平均，被噪声标注的片段不参与平均
# 但需要验证1s是否太短
# seg_length=5, (276, 5) -> (23, 5)
def get_pe_clean(eeg_data, seg_length, noise_idx):
    
    pe = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            pe[i,j] = ant.perm_entropy(eeg_data[i,j,:], normalize=True)
    print('pe.shape', pe.shape)
    pe_1m = noise_handle(pe, seg_length, noise_idx)
    for i in range(pe.shape[1]):
        plt.plot(pe[:,i])
    plt.show()
    for i in range(pe_1m.shape[1]):
        plt.plot(pe_1m[:,i])
    plt.show()
    return pe_1m

def get_lzc_clean(eeg_data, seg_length, noise_idx):
    lzc = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[0]):
        for j in range(eeg_data.shape[1]):
            x_bin = eeg_data[i,j,:].copy()*0
            thr = np.mean(eeg_data[i,j,:])
            x_bin[np.where(eeg_data[i,j,:] >= thr)] = 1
            lzc[i,j] = ant.lziv_complexity(x_bin, normalize=True)
    print('lzc.shape', lzc.shape)
    lzc_1m = noise_handle(lzc, seg_length, noise_idx)
    for i in range(lzc.shape[1]):
        plt.plot(lzc[:,i])
    plt.show()
    for i in range(lzc_1m.shape[1]):
        plt.plot(lzc_1m[:,i])
    plt.show()
    return lzc_1m

def get_svd_clean(eeg_data, seg_length, noise_idx, fs):
    freqs_num = 4
    f_start = [1,4,8,14]
    f_end = [4,8,14,30]
    svd = np.zeros(((freqs_num, eeg_data.shape[0])))
    for i in range(eeg_data.shape[0]):
        eeg_tmp = eeg_data[i,:,:][np.newaxis, :, :]
        for j in range(freqs_num):
            conn = mne_connectivity.spectral_connectivity_time(eeg_tmp, [f_start[j], f_end[j]], method='coh', 
                                                               sfreq=fs, faverage=True)
            print('conn.shape', conn.shape)
            conn = np.squeeze(conn.get_data()[0])
            conn_matrix = conn.reshape((eeg_data.shape[1], -1))
            conn_matrix[np.triu_indices(eeg_data.shape[1], 1)] = conn_matrix[np.tril_indices(eeg_data.shape[1], -1)]
            conn_matrix[np.diag_indices(eeg_data.shape[1])] = 1
            s, _ = np.linalg.eig(conn_matrix)
            svd[j,i] = np.abs(np.max(s))
    print('svd.shape', svd.shape)
    svd_1m = noise_handle_svd(svd, seg_length, noise_idx)
    for i in range(svd.shape[0]):
        plt.plot(svd[i,:])
    plt.show()
    for i in range(svd_1m.shape[0]):
        plt.plot(svd_1m[i,:])
    plt.show()
    return svd_1m

def get_bispectrum(eeg_data, seg_length):
    L = 1
    # seg_length = N/fs
    fre_low_idx = int(seg_length * 1)
    fre_high_idx = int(seg_length * 30)
    w = hamming(eeg_data.shape[-1])
    ywf = fft(w * detrend(eeg_data, type='constant'))
    print('ywf.shape', ywf.shape)
    ywf = ywf[:,:,fre_low_idx:fre_high_idx]
    print('ywf.shape', ywf.shape)
    bispectrum = np.zeros((ywf.shape[0], ywf.shape[1], ywf.shape[2]//2, ywf.shape[2]//2))
    for i in range(bispectrum.shape[-2]):
        for j in range(bispectrum.shape[-1]):
            bispectrum[:,:,i,j] = np.abs(ywf[:,:,i] * ywf[:,:,j] * np.conj(ywf[:,:,i+j]))
    return bispectrum

def get_bism_pe(eeg_data, seg_length):
    bism = get_bispectrum(eeg_data, seg_length)
    pe_first = np.zeros(bism.shape[0:-1])
    for i in range(bism.shape[0]):
        for j in range(bism.shape[1]):
            for k in range(bism.shape[2]):
                pe_first[i,j,k] = ant.perm_entropy(bism[i,j,k,:], normalize=True)
    pe_second = np.zeros(pe_first.shape[0:-1])
    for i in range(pe_first.shape[0]):
        for j in range(pe_first.shape[1]):
            pe_second[i,j] = ant.perm_entropy(pe_first[i,j,:], normalize=True)
    print('pe_second.shape', pe_second.shape)
    for i in range(pe_second.shape[1]):
        plt.plot(pe_second[:,i])
    plt.show()
    return pe_second   