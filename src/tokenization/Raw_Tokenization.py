import numpy as np
import matplotlib.pyplot as plt
from .Base_Tokenizer import BaseTokenizer

class RawTokenizer(BaseTokenizer):
    def __init__(self, subject_id, model_weights='', redo_tokenization=False):
        super().__init__(subject_id, model_weights, redo_tokenization)

    def fit(self, train_input_signal, train_target_signal, input_token_size, target_token_size,
            input_sample_rate, target_sample_rate, input_channel_names, target_channel_names,
            input_t_min=0, input_t_max=1, target_t_min=0, target_t_max=1):
        self.input_window_size = input_token_size
        self.target_window_size = target_token_size
        self.input_channel_names = input_channel_names
        self.target_channel_names = target_channel_names
        self.tokenization_dict = {
            'input': {'window_size': input_token_size},
            'target': {'window_size': target_token_size}
        }

    def tokenize(self, signal_type, data):
        # For raw data, we just return the data as is
        return data

    def inverse_tokenize(self, signal_type, data):
        # For raw data, inverse tokenization is just the identity operation
        return data

    def plot_components(self, signal_type, path=''):
        print(f"Raw tokenization does not have components to plot for {signal_type} signal.")

    def plot_explained_variance(self, signal_type, path=''):
        print(f"Raw tokenization does not have explained variance to plot for {signal_type} signal.")