'''
Functional Principal Component Analysis (FPCA) for EEG and NIRS data
'''

import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import concurrent.futures

from processing.Format_Data import grab_ordered_windows

def fdarray_to_numpy(data):
    '''Convert fdarray to numpy array
    input:
        fdarray (samples x window)
    output:
        data (samples x channels x window)
    '''
    n_samples, window, _ = data.data_matrix.shape
    data_array = data.data_matrix
    data_array = data_array.reshape(n_samples, window)
    return data_array

def numpy_to_fdarray(data):
    '''Convert numpy array to fdarray
    input:
        data (samples x channels x window)
    output:
        fdarray (samples x window)
    '''
    n_samples, window = data.shape
    fdarray = skfda.FDataGrid(data, grid_points=np.arange(window))
    return fdarray

def inverse_transform_fpca_over_channels(data, fpca_dict, window):
    '''Perform FPCA over channels of the data
    input:
        data (samples x channels x tokens)
        fpca_dict (dictionary over channels of fit fpca object)
    output:
        data (samples x channels x window)
    '''
    n_samples, n_channels, _ = data.shape

    tokenized_data = np.zeros((n_samples, n_channels, window))
    for i in range(n_channels):
        fpca = fpca_dict[i]
        inverse_transform_fdarray = fpca.inverse_transform(data[:, i, :])
        tokenized_data[:, i, :] = fdarray_to_numpy(inverse_transform_fdarray)

    return tokenized_data

def perform_fpca_over_channels(data, fpca_dict, n_components):
    '''Perform FPCA over channels of the data
    input:
        data (samples x channels x window)
        fpca_dict (dictionary over channels of fit fpca object)
    output:
        data (samples x channels x tokens)
    '''
    n_samples, n_channels, _ = data.shape
    tokenized_data = np.zeros((n_samples, n_channels, n_components))
    for i in range(n_channels):
        fpca = fpca_dict[i]
        fdarray = numpy_to_fdarray(data[:, i, :])
        print(data.shape)
        print(fdarray.shape)
        print(n_components)
        # print fpca size
        print
        tokenized_data[:, i, :] = fpca.transform(fdarray)

    return tokenized_data

def fit_fpca_single_channel(data, n_components, channel_idx):
    '''Fit PCA model to the data
    input:
        data (samples x window)
        n_components (int)
    output:
        pca (fit pca object)
    '''
    fdarray = numpy_to_fdarray(data)
    fpca = FPCA(n_components=n_components)
    fpca.fit(fdarray)
    return channel_idx, fpca

def fit_fpca_model(data, n_components):
    '''Fit PCA model to the data
    input:
        data (samples x channels x window)
        n_components (int)
    output:
        pca_dict (dictionary over channels of fit pca object)
    '''
    n_samples, n_channels, _ = data.shape
    pca_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_fpca_single_channel, data[:, i, :], n_components, i)
            for i in range(n_channels)
        ]
        for future in concurrent.futures.as_completed(futures):
            channel_idx, pca = future.result()
            pca_dict[channel_idx] = pca
            if channel_idx % 10 == 0:
                print(f'Finished fitting PCA for channel {channel_idx + 1}')

    return pca_dict

def plot_explained_variance_over_dict(fpca_dict, channel_names, path=''):
    '''Plot explained variance over channels
    input:
        fpca_dict (dictionary over channels of fit fpca object)
    '''
    explained_variance = []

    fig, axs = plt.subplots(len(fpca_dict), 1, figsize=(10, 50))
    for i in range(len(fpca_dict)):
        channel_name = channel_names[i]
        variance_list = fpca_dict[i].explained_variance_ratio_
        # plot bar of percentages
        axs[i].bar(np.arange(len(variance_list)), variance_list)
        axs[i].text(0.5, 0.9, f'{channel_name}', horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
    if len(path) > 0:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def get_fpca_dict(subject_id, 
                  train_nirs_data, 
                  train_eeg_data, 
                  nirs_token_size, 
                  eeg_token_size,
                  model_weights='',
                  nirs_t_min=0,
                  nirs_t_max=1,
                  eeg_t_min=0,
                  eeg_t_max=1):
    fpca_dict_path = os.path.join(model_weights, f'fpca_dict_{subject_id}.pkl')
    if not os.path.exists(fpca_dict_path):
        print(f'Building FPCA Dict')
        eeg_windowed_train, nirs_windowed_train, meta_data = grab_ordered_windows(
                    nirs_data=train_nirs_data, 
                    eeg_data=train_eeg_data,
                    sampling_rate=200,
                    nirs_t_min=nirs_t_min, 
                    nirs_t_max=nirs_t_max,
                    eeg_t_min=eeg_t_min, 
                    eeg_t_max=eeg_t_max)
        print(f'EEG FPCA Shape: {eeg_windowed_train.shape}')
        print(f'NIRS FPCA Shape: {nirs_windowed_train.shape}')
        eeg_fpca_dict = fit_fpca_model(eeg_windowed_train, eeg_token_size)
        nirs_fpca_dict = fit_fpca_model(nirs_windowed_train, nirs_token_size)
        fpca_dict = {'eeg': eeg_fpca_dict, 'nirs': nirs_fpca_dict}
        joblib.dump(fpca_dict, fpca_dict_path)
    else:
        fpca_dict = joblib.load(fpca_dict_path)
    return fpca_dict