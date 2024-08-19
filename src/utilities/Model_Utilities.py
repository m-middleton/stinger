'''
    This file contains utility functions to create and predict using different models
'''

import numpy as np

import torch
import torch.nn as nn

from iTransformer.iTransformer.iTransformerTranscoding import iTransformer
from processing.FPCA import inverse_transform_fpca_over_channels

class LSTMModel(nn.Module):
    def __init__(self, input_features, hidden_dim, output_steps):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_steps)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out)
        return y_pred

def create_rnn(n_input, n_output):
    '''
       n_input (360)
       n_output (150)
    '''
    print(f'Creating RNN with input size: {n_input} and output size: {n_output}')
    model = LSTMModel(n_input, 20, n_output)

    return model

def create_mlp(n_input, n_output):
    '''
       n_input (360)
       n_output (150)
    '''
    print(f'Creating MLP with input size: {n_input} and output size: {n_output}')
    model = nn.Sequential(
        nn.Linear(n_input, 256),
        nn.ReLU(),
        nn.Linear(256, n_output)
    )

    return model

def create_transformer(nirs_channels_to_use_base, eeg_channels_to_use, fnirs_lookback, eeg_lookback):
    model = iTransformer(
            num_variates = len(nirs_channels_to_use_base),
            lookback_len = fnirs_lookback,      # or the lookback length in the paper
            target_num_variates=len(eeg_channels_to_use),
            target_lookback_len=eeg_lookback,
            dim = 256,                          # model dimensions
            depth = 6,                          # depth
            heads = 8,                          # attention heads
            dim_head = 64,                      # head dimension
            attn_dropout=0.1,
            ff_mult=4,
            ff_dropout=0.1,
            num_mem_tokens=10,
            num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
            use_reversible_instance_norm = True # use reversible instance normalization, proposed here https://openreview.net/forum?id=cGDAkQo1C0p . may be redundant given the layernorms within iTransformer (and whatever else attention learns emergently on the first layer, prior to the first layernorm). if i come across some time, i'll gather up all the statistics across variates, project them, and condition the transformer a bit further. that makes more sense
        )
    
    return model

def predict_eeg(model, data_loader, n_samples, n_channels, n_lookback, eeg_token_size, eeg_fpcas):
    # Set model to evaluation mode
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference on test data
    predictions = []
    targets = []
    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float()
        predictions.append(model(X_batch).detach().cpu().numpy())
        targets.append(y_batch.detach().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    targets = targets.reshape((n_samples, n_lookback, n_channels))

    # inverse CA on predictions
    predictions = predictions.reshape(n_samples, eeg_token_size, n_channels)
    predictions = predictions.transpose(0,2,1)
    # predictions = predictions.reshape(eeg_windowed_test.shape[0], eeg_token_size, eeg_windowed_test.shape[2])
    predictions = inverse_transform_fpca_over_channels(predictions, eeg_fpcas, n_lookback)
    predictions = predictions.transpose(0,2,1)

    return targets, predictions