'''
Functional Principal Component Analysis (FPCA) for generic input and target signals
'''

from .Base_Tokenizer import BaseTokenizer

import skfda
from skfda.preprocessing.dim_reduction import FPCA

from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import concurrent.futures

from processing.Format_Data import grab_ordered_windows  # Ensure this is generalized

def fdarray_to_numpy(data):
    '''Convert fdarray to numpy array
    input:
        fdarray (samples x window)
    output:
        data (samples x window)
    '''
    n_samples, window, _ = data.data_matrix.shape
    data_array = data.data_matrix
    data_array = data_array.reshape(n_samples, window)
    return data_array

def numpy_to_fdarray(data):
    '''Convert numpy array to fdarray
    input:
        data (samples x window)
    output:
        fdarray (samples x window)
    '''
    n_samples, window = data.shape
    fdarray = skfda.FDataGrid(data, grid_points=np.arange(window))
    return fdarray

def fit_fpca_single_channel(data, n_components, channel_name, channel_idx):
    '''Fit FPCA model to the data for a single channel
    input:
        data (samples x window)
        n_components (int)
    output:
        channel_idx, channel_name, fpca (fit FPCA object)
    '''
    fdarray = numpy_to_fdarray(data)
    fpca = FPCA(n_components=n_components)
    fpca.fit(fdarray)
    return channel_idx, channel_name, fpca

def fit_fpca_model(data, n_components, channel_names):
    '''Fit FPCA model to the data across all channels
    input:
        data (samples x channels x window)
        n_components (int)
    output:
        fpca_dict (dictionary over channels of fit FPCA object)
    '''
    n_samples, n_channels, _ = data.shape
    fpca_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_fpca_single_channel, data[:, idx, :], n_components, channel_name, idx)
            for idx, channel_name in enumerate(channel_names)
        ]
        for future in concurrent.futures.as_completed(futures):
            channel_idx, channel_name, fpca = future.result()
            fpca_dict[channel_name] = fpca
            if (channel_idx + 1) % 10 == 0:
                print(f'Finished fitting FPCA for channel {channel_idx + 1}')

    return fpca_dict

class MultiChannelFPCA(BaseTokenizer):
    def __init__(self, subject_id, model_weights='', redo_tokenization=False):
        super().__init__(subject_id, model_weights, redo_tokenization)

    def fit(self, train_input_signal, train_target_signal, input_token_size, target_token_size,
            input_sample_rate, target_sample_rate, input_channel_names, target_channel_names,
            input_t_min=0, input_t_max=1, target_t_min=0, target_t_max=1):
        fpca_dict_path = os.path.join(self.model_weights, f'fpca_dict_{self.subject_id}.pkl')
        
        self.input_window_size = (input_t_max + abs(input_t_min)) * input_sample_rate
        self.target_window_size = (target_t_max + abs(target_t_min)) * target_sample_rate
        self.input_channel_names = input_channel_names
        self.target_channel_names = target_channel_names
        self.input_token_size = input_token_size
        self.target_token_size = target_token_size

        if not os.path.exists(fpca_dict_path) or self.redo_tokenization:
            print('Building FPCA Dict')
            input_windowed_train, target_windowed_train, _, _ = grab_ordered_windows(
                input_signal=train_input_signal, 
                target_signal=train_target_signal,
                input_sample_rate=input_sample_rate,
                target_sample_rate=target_sample_rate,
                input_t_min=input_t_min, 
                input_t_max=input_t_max,
                target_t_min=target_t_min,
                target_t_max=target_t_max)

            self.tokenization_dict['input'] = fit_fpca_model(input_windowed_train, input_token_size, self.input_channel_names)
            self.tokenization_dict['target'] = fit_fpca_model(target_windowed_train, target_token_size, self.target_channel_names)
            joblib.dump(self.tokenization_dict, fpca_dict_path)
        else:
            self.tokenization_dict = joblib.load(fpca_dict_path)

        return self

    def tokenize(self, signal_type, data):
        return self.perform_fpca_over_channels(signal_type, data)

    def inverse_tokenize(self, signal_type, data):
        if signal_type == 'target':
            # Transpose data to match expected input shape
            data = data.transpose(0, 2, 1)
        untokenized_data = self.inverse_transform_fpca_over_channels(data, signal_type)
        if signal_type == 'target':
            # Transpose back to original shape
            untokenized_data = untokenized_data.transpose(0, 2, 1)
        return untokenized_data

    def plot_components(self, signal_type, path=''):
        self.plot_shape(signal_type, path)

    def plot_explained_variance(self, signal_type, path=''):
        self.plot_explained_variance_over_dict(signal_type, path)

    def plot__fpca_shape(self, signal_type, path=''):
        '''Plot the shape of FPCA components in 3D for each channel
        input:
            signal_type (str): 'input' or 'target'
            path (str): path to save the plot (optional)
        '''
        fpca_dict = self.tokenization_dict[signal_type]
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names

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

    def plot_explained_variance_over_dict(self, signal_type, path=''):
        '''Plot explained variance over channels
        input:
            signal_type (str): 'input' or 'target'
            path (str): path to save the plot (optional)
        '''
        fpca_dict = self.tokenization_dict[signal_type]
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names

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

    def perform_fpca_over_channels(self, signal_type, data):
        '''Perform FPCA over channels of the data
        input:
            data (samples x channels x window)
            signal_type (str): 'input' or 'target'
        output:
            tokenized_data (samples x channels x tokens)
        '''
        fpca_dict = self.tokenization_dict[signal_type]
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names
        n_components = self.input_token_size if signal_type == 'input' else self.target_token_size

        n_samples, n_channels, _ = data.shape
        tokenized_data = np.zeros((n_samples, n_channels, n_components))
        for idx, channel_name in enumerate(channel_names):
            fpca = fpca_dict[channel_name]
            fdarray = numpy_to_fdarray(data[:, idx, :])
            tokenized_data[:, idx, :] = fpca.transform(fdarray)

        return tokenized_data
    
    def inverse_transform_fpca_over_channels(self, data, signal_type):
        '''Inverse transform the FPCA over channels
        input:
            data (samples x channels x tokens)
            signal_type (str): 'input' or 'target'
        output:
            untokenized_data (samples x channels x window)
        '''
        fpca_dict = self.tokenization_dict[signal_type]
        window_size = self.input_window_size if signal_type == 'input' else self.target_window_size
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names

        n_samples, n_channels, _ = data.shape

        untokenized_data = np.zeros((n_samples, n_channels, int(window_size)))
        for idx, channel_name in enumerate(channel_names):
            fpca = fpca_dict[channel_name]
            inverse_transform_fdarray = fpca.inverse_transform(data[:, idx, :])
            untokenized_data[:, idx, :] = fdarray_to_numpy(inverse_transform_fdarray)

        return untokenized_data
