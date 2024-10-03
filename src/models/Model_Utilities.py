'''
    This file contains utility functions to create and predict using different models
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from iTransformer.iTransformer.iTransformerTranscoding import iTransformer

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_channels):
        super().__init__()
        self.embedding = nn.Linear(3, d_model)
        self.num_channels = num_channels

    def forward(self, coordinates):
        # coordinates shape: (num_channels, 3)
        return self.embedding(coordinates)  # (num_channels, d_model)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_features, 
                 output_features, 
                 num_nirs_channels, 
                 num_eeg_channels, 
                 hidden_dim, 
                 nirs_seq_len, 
                 eeg_seq_len, 
                 eeg_position_hidden_dimension=2):
        super(EnhancedLSTMModel, self).__init__()
        self.nirs_positional_encoding = PositionalEncoding(input_features // num_nirs_channels, num_nirs_channels)
        self.eeg_positional_encoding = PositionalEncoding(eeg_position_hidden_dimension, num_eeg_channels)
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim + eeg_position_hidden_dimension * num_eeg_channels, output_features)
        self.num_nirs_channels = num_nirs_channels
        self.num_eeg_channels = num_eeg_channels
        self.nirs_seq_len = nirs_seq_len
        self.eeg_seq_len = eeg_seq_len
        self.hidden_dim = hidden_dim
    
    def forward(self, x, nirs_coordinates, eeg_coordinates):
        # x shape: (batch_size, seq_len, num_nirs_channels * feature_dim)
        # nirs_coordinates shape: (num_nirs_channels, 3)
        # eeg_coordinates shape: (num_eeg_channels, 3)
        batch_size, seq_len, _ = x.shape
        
        # Add NIRS positional encoding
        nirs_pos_encoding = self.nirs_positional_encoding(nirs_coordinates)
        nirs_pos_encoding = nirs_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        nirs_pos_encoding = nirs_pos_encoding.reshape(batch_size, self.nirs_seq_len, self.num_nirs_channels)

        x_with_pos = x + nirs_pos_encoding

        # flatten the input
        x_with_pos = x_with_pos.reshape(batch_size, -1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x_with_pos)
        
        # Add EEG positional encoding
        eeg_pos_encoding = self.eeg_positional_encoding(eeg_coordinates)
        eeg_pos_encoding = eeg_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine LSTM output with EEG positional encoding
        lstm_out = lstm_out.view(batch_size, self.hidden_dim)
        eeg_pos_encoding = eeg_pos_encoding.view(batch_size, -1)
        combined = torch.cat([lstm_out, eeg_pos_encoding], dim=-1)
        
        # Final prediction
        y_pred = self.linear(combined)
        return y_pred

def create_rnn(n_input, 
               n_output,  
               nirs_sequence_length,
               eeg_sequence_length, 
               num_nirs_channels, 
               num_eeg_channels,
               hidden_dim=64):
    print(f'Creating Enhanced RNN with input features: {n_input} and output features: {n_output}')
    model = EnhancedLSTMModel(n_input, n_output, num_nirs_channels, num_eeg_channels, hidden_dim, nirs_sequence_length, eeg_sequence_length)
    return model

class MLPModel(nn.Module):
    def __init__(self, n_input, n_output, hidden_dims, dropout_rate, activation):
        super(MLPModel, self).__init__()
        
        layers = []
        in_features = n_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                activation(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, n_output))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, nirs_coordinates, eeg_coordinates):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)

def create_mlp(n_input, 
               n_output,
               nirs_sequence_length,
               eeg_sequence_length, 
               num_nirs_channels, 
               num_eeg_channels,
               hidden_dims=[512, 256, 128],
               dropout_rate=0.2,
               activation=nn.ReLU):
    '''
    Creates a Multi-Layer Perceptron (MLP) for transcoding.
    
    Args:
        n_input (int): Number of input features (e.g., 360)
        n_output (int): Number of output features (e.g., 150)
        num_nirs_channels (int): Number of NIRS channels
        num_eeg_channels (int): Number of EEG channels
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        activation (nn.Module): Activation function to use
    
    Returns:
        MLPModel: The MLP model
    '''
    print(f'Creating MLP with input size: {n_input} and output size: {n_output}')
    
    model = MLPModel(n_input, n_output, hidden_dims, dropout_rate, activation)

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

def predict_eeg(model, data_loader, spatial_bias, nirs_coordinates, eeg_coordinates, n_samples, n_channels, n_lookback, eeg_token_size, tokenizer):
    # Set model to evaluation mode
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            spatial_bias_batch = spatial_bias.to(device)
            nirs_coordinates = nirs_coordinates.to(device)
            eeg_coordinates = eeg_coordinates.to(device)
            
            # prediction = model(X_batch, spatial_bias_batch)
            prediction = model(X_batch, nirs_coordinates, eeg_coordinates)

            predictions.append(prediction.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # predictions = []
    # targets = []
    # for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
    #     X_batch = X_batch.to(device).float()
    #     y_batch = y_batch.to(device).float()
    #     predictions.append(model(X_batch).detach().cpu().numpy())
    #     targets.append(y_batch.detach().cpu().numpy())

    # predictions = np.array(predictions)
    # targets = np.array(targets)
    
    targets = targets.reshape((n_samples, n_lookback, n_channels))
    predictions = predictions.reshape(n_samples, eeg_token_size, n_channels)

    # inverse CA on predictions
    predictions = tokenizer.inverse_tokenizer(predictions)

    return targets, predictions