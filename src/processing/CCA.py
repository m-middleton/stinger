'''
This file contains functions to perform Canonical Correlation Analysis (CCA) on the data.
'''

import concurrent.futures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

import numpy as np
import os
import joblib

from processing.Format_Data import grab_ordered_windows

def inverse_transform_cca_over_channels(data, cca_dict, window):
    '''Perform CCA over channels of the data
    input:
        data (samples x channels x tokens)
        cca_dict (dictionary over channels of fit cca object)
    output:
        data (samples x channels x window)
    '''
    n_samples, n_channels, _ = data.shape

    tokenized_data = np.zeros((n_samples, n_channels, window))
    for i in range(n_channels):
        cca = cca_dict[i]
        tokenized_data[:, i, :] = cca.inverse_transform(data[:, i, :])

    return tokenized_data

def perform_cca_over_channels(data, cca_dict, n_components):
    '''Perform CCA over channels of the data
    input:
        data (samples x channels x window)
        cca_dict (dictionary over channels of fit cca object)
    output:
        data (samples x channels x tokens)
    '''
    n_samples, n_channels, _ = data.shape
    tokenized_data = np.zeros((n_samples, n_channels, n_components))
    for i in range(n_channels):
        cca = cca_dict[i]
        tokenized_data[:, i, :] = cca.transform(data[:, i, :])

    return tokenized_data

def fit_cca_single_channel(data_reshaped_a, data_reshaped_b, n_components, channel_idx):
    cca = CCA(n_components=n_components)
    cca.fit(data_reshaped_b, data_reshaped_a)
    return channel_idx, cca

def fit_cca_model(time_series_a, time_series_b, n_components):
    '''Fit CCA model to the data
    input:
        time_series_a (samples x channels x window)
        time_series_b (samples x channels x window)
        n_components (int)
    output:
        cca_dict (dictionary over channels of fit cca object)
    '''
    n_samples, n_channels, _ = time_series_b.shape
    cca_dict = {}

    data_reshaped_a = time_series_a.reshape(n_samples, -1)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_cca_single_channel, data_reshaped_a, time_series_b[:, i, :], n_components, i)
            for i in range(n_channels)
        ]
        for future in concurrent.futures.as_completed(futures):
            channel_idx, cca = future.result()
            cca_dict[channel_idx] = cca
            if channel_idx % 10 == 0:
                print(f'Finished fitting CCA for channel {channel_idx + 1}')

    return cca_dict

def get_cca_dict(subject_id, 
                 train_nirs_data, 
                 train_eeg_data, 
                 token_size,
                 model_weights,
                 nirs_t_min=0,
                 nirs_t_max=1):
    cca_dict_path = os.path.join(model_weights, f'cca_dict_{subject_id}.pkl')
    if not os.path.exists(cca_dict_path):
        print(f'Building CCA Dict')
        eeg_windowed_train, nirs_windowed_train, meta_data = grab_ordered_windows(
                    nirs_data=train_nirs_data, 
                    eeg_data=train_eeg_data,
                    sampling_rate=200,
                    nirs_t_min=nirs_t_min, 
                    nirs_t_max=nirs_t_max,
                    eeg_t_min=0, 
                    eeg_t_max=1)
        print(f'EEG CCA Shape: {eeg_windowed_train.shape}')
        print(f'NIRS CCA Shape: {nirs_windowed_train.shape}')
        cca_dict = fit_cca_model(eeg_windowed_train, nirs_windowed_train, token_size)
        joblib.dump(cca_dict, cca_dict_path)
    else:
        cca_dict = joblib.load(cca_dict_path)

    return cca_dict
