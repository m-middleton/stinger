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

from sklearn.manifold import TSNE

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

def fit_fpca_single_channel(data, n_components, channel_name, channel_idx):
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
    return channel_idx, channel_name, fpca

def fit_fpca_model(data, n_components, channel_names):
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
            executor.submit(fit_fpca_single_channel, data[:, idx, :], n_components, channel_name, idx)
            for idx, channel_name in enumerate(channel_names)
        ]
        for future in concurrent.futures.as_completed(futures):
            channel_idx, channel_name, pca = future.result()
            pca_dict[channel_name] = pca
            if channel_idx % 10 == 0:
                print(f'Finished fitting PCA for channel {channel_idx+1}')

    return pca_dict



class MultiChannelFPCA:
    def __init__(self, 
                 subject_id, 
                  train_nirs_data, 
                  train_eeg_data, 
                  nirs_token_size, 
                  eeg_token_size,
                  fnirs_sample_rate,
                  eeg_sample_rate,
                  nirs_channel_names,
                  eeg_channel_names,
                  model_weights='',
                  nirs_t_min=0,
                  nirs_t_max=1,
                  eeg_t_min=0,
                  eeg_t_max=1):
        fpca_dict_path = os.path.join(model_weights, f'fpca_dict_{subject_id}.pkl')
        
        self.window_size = (eeg_t_max + abs(eeg_t_min)) * eeg_sample_rate
        self.nirs_channel_names = nirs_channel_names
        self.eeg_channel_names = eeg_channel_names

        if True: #not os.path.exists(fpca_dict_path):
            print(f'Building FPCA Dict')
            eeg_windowed_train, nirs_windowed_train, _, _ = grab_ordered_windows(
                nirs_data=train_nirs_data, 
                eeg_data=train_eeg_data,
                nirs_sampling_rate=fnirs_sample_rate,
                eeg_sampling_rate=eeg_sample_rate,
                nirs_t_min=nirs_t_min,
                nirs_t_max=nirs_t_max,
                eeg_t_min=eeg_t_min, 
                eeg_t_max=eeg_t_max)
            
            print(f'EEG FPCA Shape: {eeg_windowed_train.shape}')
            print(f'NIRS FPCA Shape: {nirs_windowed_train.shape}')
            eeg_fpca_dict = fit_fpca_model(eeg_windowed_train, eeg_token_size, eeg_channel_names)
            nirs_fpca_dict = fit_fpca_model(nirs_windowed_train, nirs_token_size, nirs_channel_names)
            fpca_dict = {'eeg': eeg_fpca_dict, 'nirs': nirs_fpca_dict}
            joblib.dump(fpca_dict, fpca_dict_path)
        else:
            fpca_dict = joblib.load(fpca_dict_path)

        self.all_fpca_dict = fpca_dict

    def plot_shape(self, dict_name, path=''):
        '''Plot the shape of PCA components in 3D for each channel
        input:
            dict_name (str): 'eeg' or 'nirs'
            path (str): path to save the plot (optional)
        '''
        fpca_dict = self.all_fpca_dict[dict_name]
        channel_names = self.eeg_channel_names if dict_name == 'eeg' else self.nirs_channel_names

        n_channels = len(channel_names)
        n_cols = 3
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for idx, channel_name in enumerate(channel_names):
            fpca = fpca_dict[channel_name]
            components = fpca.components_

            # Convert FDataGrid to numpy array if necessary
            if isinstance(components, skfda.representation.grid.FDataGrid):
                components = components.data_matrix.squeeze()

            # Create 3D subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

            # Plot the first three principal components
            if components.shape[0] >= 3:
                ax.scatter(components[0], components[1], components[2])
            elif components.shape[0] == 2:
                ax.scatter(components[0], components[1], np.zeros_like(components[0]))
            elif components.shape[0] == 1:
                ax.scatter(components[0], np.zeros_like(components[0]), np.zeros_like(components[0]))

            ax.set_title(f'{channel_name}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')

        plt.tight_layout()
        
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def plot_explained_variance_over_dict(self, dict_name, path=''):
        '''Plot explained variance over channels
        input:
            dict_name (str): 'eeg' or 'nirs'
            path (str): path to save the plot (optional)
        '''
        fpca_dict = self.all_fpca_dict[dict_name]
        channel_names = self.eeg_channel_names if dict_name == 'eeg' else self.nirs_channel_names

        n_channels = len(channel_names)
        n_cols = 3
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for idx, channel_name in enumerate(channel_names):
            variance_list = fpca_dict[channel_name].explained_variance_ratio_
            
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            ax.bar(np.arange(len(variance_list)), variance_list)
            ax.set_title(f'{channel_name}')
            ax.set_xlabel('Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_ylim(0, 1)

        plt.tight_layout()
        
        if path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def perform_fpca_over_channels(self, dict_name, data, n_components):
        '''Perform FPCA over channels of the data
        input:
            data (samples x channels x window)
            fpca_dict (dictionary over channels of fit fpca object)
        output:
            data (samples x channels x tokens)
        '''
        fpca_dict = self.all_fpca_dict[dict_name]
        channel_names = self.eeg_channel_names if dict_name == 'eeg' else self.nirs_channel_names

        n_samples, n_channels, _ = data.shape
        tokenized_data = np.zeros((n_samples, n_channels, n_components))
        for idx, channel_name in enumerate(channel_names):
            fpca = fpca_dict[channel_name]
            fdarray = numpy_to_fdarray(data[:, idx, :])
            tokenized_data[:, idx, :] = fpca.transform(fdarray)

        return tokenized_data
    
    def inverse_transform_fpca_over_channels(self, data, dict_name):
        '''Perform FPCA over channels of the data
        input:
            data (samples x channels x tokens)
            fpca_dict (dictionary over channels of fit fpca object)
        output:
            data (samples x channels x window)
        '''
        fpca_dict = self.all_fpca_dict[dict_name]
        channel_names = self.eeg_channel_names if dict_name == 'eeg' else self.nirs_channel_names

        n_samples, n_channels, _ = data.shape

        untokenized_data = np.zeros((n_samples, n_channels, self.window_size))
        for idx, channel_name in enumerate(channel_names):
            fpca = fpca_dict[channel_name]
            inverse_transform_fdarray = fpca.inverse_transform(data[:, idx, :])
            untokenized_data[:, idx, :] = fdarray_to_numpy(inverse_transform_fdarray)

        return untokenized_data

    def inverse_tokenizer(self, data):
        '''
        input:
            data (samples x tokens x channels)
        output:
            data (samples x window x channels
        '''
        # inverse CA on predictions
        data = data.transpose(0,2,1)
        untokenized_data = self.inverse_transform_fpca_over_channels(data, 'eeg')
        untokenized_data = untokenized_data.transpose(0,2,1)

        return untokenized_data
