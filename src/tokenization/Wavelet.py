import os
import numpy as np
import pywt
import joblib
from .Base_Tokenizer import BaseTokenizer
from processing.Format_Data import grab_ordered_windows

def fit_wavelet_model(data, n_components, channel_names):
    '''Fit wavelet model to the data across all channels
    input:
        data (samples x channels x window)
        n_components (int)
    output:
        wavelet_dict (dictionary over channels of wavelet coefficients)
    '''
    n_samples, n_channels, _ = data.shape
    wavelet_dict = {}

    for idx, channel_name in enumerate(channel_names):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data[:, idx, :], 'db4', level=pywt.dwt_max_level(data.shape[2], 'db4'))
        
        # Concatenate coefficients and truncate to n_components
        flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])[:n_components]
        
        wavelet_dict[channel_name] = flattened_coeffs

        if (idx + 1) % 10 == 0:
            print(f'Finished fitting wavelet for channel {idx + 1}')

    return wavelet_dict

class MultiChannelWavelet(BaseTokenizer):
    def __init__(self, subject_id, model_weights='', redo_tokenization=False):
        super().__init__(subject_id, model_weights, redo_tokenization)

    def fit(self, train_input_signal, train_target_signal, input_token_size, target_token_size,
            input_sample_rate, target_sample_rate, input_channel_names, target_channel_names,
            input_t_min=0, input_t_max=1, target_t_min=0, target_t_max=1):
        wavelet_dict_path = os.path.join(self.model_weights, f'wavelet_dict_{self.subject_id}.pkl')
        
        self.input_window_size = int((input_t_max + abs(input_t_min)) * input_sample_rate)
        self.target_window_size = int((target_t_max + abs(target_t_min)) * target_sample_rate)
        self.input_channel_names = input_channel_names
        self.target_channel_names = target_channel_names
        self.input_token_size = input_token_size
        self.target_token_size = target_token_size

        if not os.path.exists(wavelet_dict_path) or self.redo_tokenization:
            print('Building Wavelet Dict')
            input_windowed_train, target_windowed_train, _, _ = grab_ordered_windows(
                input_signal=train_input_signal, 
                target_signal=train_target_signal,
                input_sample_rate=input_sample_rate,
                target_sample_rate=target_sample_rate,
                input_t_min=input_t_min, 
                input_t_max=input_t_max,
                target_t_min=target_t_min,
                target_t_max=target_t_max)

            self.tokenization_dict['input'] = fit_wavelet_model(input_windowed_train, input_token_size, self.input_channel_names)
            self.tokenization_dict['target'] = fit_wavelet_model(target_windowed_train, target_token_size, self.target_channel_names)
            joblib.dump(self.tokenization_dict, wavelet_dict_path)
        else:
            self.tokenization_dict = joblib.load(wavelet_dict_path)

        return self

    def tokenize(self, signal_type, data):
        return self.perform_wavelet_over_channels(signal_type, data)

    def inverse_tokenize(self, signal_type, data):
        if signal_type == 'target':
            # Transpose data to match expected input shape
            data = data.transpose(0, 2, 1)
        untokenized_data = self.inverse_transform_wavelet_over_channels(data, signal_type)
        if signal_type == 'target':
            # Transpose back to original shape
            untokenized_data = untokenized_data.transpose(0, 2, 1)
        return untokenized_data

    def perform_wavelet_over_channels(self, signal_type, data):
        '''Perform wavelet transform over channels of the data
        input:
            data (samples x channels x window)
            signal_type (str): 'input' or 'target'
        output:
            tokenized_data (samples x channels x tokens)
        '''
        wavelet_dict = self.tokenization_dict[signal_type]
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names
        n_components = self.input_token_size if signal_type == 'input' else self.target_token_size

        n_samples, n_channels, _ = data.shape
        tokenized_data = np.zeros((n_samples, n_channels, n_components))
        for idx, channel_name in enumerate(channel_names):
            coeffs = pywt.wavedec(data[:, idx, :], 'db4', level=pywt.dwt_max_level(data.shape[2], 'db4'))
            flattened_coeffs = np.concatenate([c.flatten() for c in coeffs])[:n_components]
            tokenized_data[:, idx, :] = flattened_coeffs

        return tokenized_data

    def inverse_transform_wavelet_over_channels(self, data, signal_type):
        '''Inverse transform the wavelet over channels
        input:
            data (samples x channels x tokens)
            signal_type (str): 'input' or 'target'
        output:
            untokenized_data (samples x channels x window)
        '''
        wavelet_dict = self.tokenization_dict[signal_type]
        window_size = self.input_window_size if signal_type == 'input' else self.target_window_size
        channel_names = self.input_channel_names if signal_type == 'input' else self.target_channel_names

        n_samples, n_channels, n_tokens = data.shape
        untokenized_data = np.zeros((n_samples, n_channels, int(window_size)))

        # Calculate expected coefficient lengths
        wavelet_levels = pywt.dwt_max_level(window_size, 'db4')
        expected_coeffs = pywt.wavedec(np.zeros(window_size), 'db4', level=wavelet_levels)
        coeff_lengths = [len(c) for c in expected_coeffs]
        total_coeffs = sum(coeff_lengths)

        for idx, channel_name in enumerate(channel_names):
            # Reconstruct coefficients
            coeffs = []
            start = 0
            
            for i, coeff_len in enumerate(coeff_lengths):
                if start >= n_tokens:
                    # If we've run out of coefficients, pad with zeros
                    coeffs.append(np.zeros((n_samples, coeff_len)))
                else:
                    # Calculate how many coefficients we can actually use
                    available_coeffs = min(coeff_len, n_tokens - start)
                    
                    if available_coeffs < coeff_len:
                        # Need to pad
                        pad_data = np.zeros((n_samples, coeff_len))
                        pad_data[:, :available_coeffs] = data[:, idx, start:start+available_coeffs]
                        coeffs.append(pad_data)
                    else:
                        # Can use full coefficients
                        coeffs.append(data[:, idx, start:start+coeff_len])
                
                start += coeff_len
            
            # Inverse wavelet transform
            reconstructed = pywt.waverec(coeffs, 'db4')
            # Ensure the output is the correct size
            untokenized_data[:, idx, :] = reconstructed[:, :int(window_size)]

        return untokenized_data

    def plot_components(self, signal_type, path=''):
        # Implement wavelet component visualization if needed
        pass

    def plot_explained_variance(self, signal_type, path=''):
        # Implement wavelet explained variance visualization if needed
        pass