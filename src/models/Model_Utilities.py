'''
    This file contains utility functions to create and predict using different models
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iTransformer.iTransformer.iTransformerTranscoding import iTransformer

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
                num_input_channels, 
                num_target_channels, 
                hidden_dim, 
                input_seq_len, 
                target_seq_len, 
                target_position_hidden_dimension=2,
                spatial_encoding=True,
                num_lstm_layers=1,
                bidirectional=False,
                dropout=0,
                use_attention=False,
                attention_heads=1):
        super(EnhancedLSTMModel, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_target_channels = num_target_channels
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        
        self.spatial_encoding = spatial_encoding
        self.use_attention = use_attention

        if self.spatial_encoding:
            self.input_positional_encoding = PositionalEncoding(input_features // num_input_channels, num_input_channels)
            self.target_positional_encoding = PositionalEncoding(target_position_hidden_dimension, num_target_channels)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.lstm = nn.LSTM(input_size=input_features, 
                            hidden_size=hidden_dim, 
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_lstm_layers > 1 else 0)
        
        if self.use_attention:
            self.attention = nn.MultiheadAttention(lstm_output_dim, attention_heads)
        
        if self.spatial_encoding:
            final_dim = lstm_output_dim + target_position_hidden_dimension * num_target_channels
        else:
            final_dim = lstm_output_dim
        
        self.linear = nn.Linear(final_dim, output_features)
    
    def forward(self, x, input_coordinates, target_coordinates):
        # x shape: (batch_size, seq_len, num_input_channels * feature_dim)
        # input_coordinates shape: (num_input_channels, 3)
        # target_coordinates shape: (num_target_channels, 3)

        batch_size, seq_len, _ = x.shape
        
        # Add input positional encoding
        if self.spatial_encoding:
            input_pos_encoding = self.input_positional_encoding(input_coordinates)
            input_pos_encoding = input_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            input_pos_encoding = input_pos_encoding.reshape(batch_size, self.input_seq_len, self.num_input_channels)

            lstm_input = x + input_pos_encoding
        else:
            lstm_input = x

        # flatten the input
        lstm_input = lstm_input.reshape(batch_size, -1)

        # Process with LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        if self.spatial_encoding:
            # Add target positional encoding
            target_pos_encoding = self.target_positional_encoding(target_coordinates)
            target_pos_encoding = target_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
            # Combine LSTM output with target positional encoding
            lstm_out = lstm_out.view(batch_size, self.hidden_dim)
            target_pos_encoding = target_pos_encoding.view(batch_size, -1)
            lstm_out = torch.cat([lstm_out, target_pos_encoding], dim=-1)

        # Final prediction
        y_pred = self.linear(lstm_out)
        return y_pred

def create_rnn(n_input, 
               n_output,  
               input_sequence_length,
               target_sequence_length, 
               num_input_channels, 
               num_target_channels,
               hidden_dim=64,
               spatial_encoding=True,
               num_lstm_layers=1,
               bidirectional=False,
               dropout=0,
               use_attention=False,
               attention_heads=1):
    print(f'Creating Enhanced RNN with input features: {n_input} and output features: {n_output}')
    model = EnhancedLSTMModel(
        input_features=n_input, 
        output_features=n_output, 
        num_input_channels=num_input_channels,
        num_target_channels=num_target_channels, 
        hidden_dim=hidden_dim, 
        input_seq_len=input_sequence_length, 
        target_seq_len=target_sequence_length,
        spatial_encoding=spatial_encoding,
        num_lstm_layers=num_lstm_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        use_attention=use_attention,
        attention_heads=attention_heads
    )
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
    
    def forward(self, x, input_coordinates, target_coordinates):
        x = x.reshape(x.shape[0], -1)
        return self.model(x)

def create_mlp(n_input, 
               n_output,
               input_sequence_length,
               target_sequence_length, 
               num_input_channels, 
               num_target_channels,
               hidden_dims=[512, 256, 128],
               dropout_rate=0.2,
               activation=nn.ReLU):
    '''
    Creates a Multi-Layer Perceptron (MLP) for transcoding.
    
    Args:
        n_input (int): Number of input features
        n_output (int): Number of output features
        num_input_channels (int): Number of input signal channels
        num_target_channels (int): Number of target signal channels
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        activation (nn.Module): Activation function to use
    
    Returns:
        MLPModel: The MLP model
    '''
    print(f'Creating MLP with input size: {n_input} and output size: {n_output}')
    
    model = MLPModel(n_input, n_output, hidden_dims, dropout_rate, activation)

    return model

def create_transformer(input_channels_to_use, target_channels_to_use, input_lookback, target_lookback):
    model = iTransformer(
            num_variates = len(input_channels_to_use),
            lookback_len = input_lookback,
            target_num_variates=len(target_channels_to_use),
            target_lookback_len=target_lookback,
            dim = 256,                          # model dimensions
            depth = 6,                          # depth
            heads = 8,                          # attention heads
            dim_head = 64,                      # head dimension
            attn_dropout=0.1,
            ff_mult=4,
            ff_dropout=0.1,
            num_mem_tokens=10,
            num_tokens_per_variate = 1,         # Experimental setting
            use_reversible_instance_norm = True # Use reversible instance normalization
        )
    
    return model

def predict_signal(model, data_loader, spatial_bias, input_coordinates, target_coordinates, n_samples, n_channels, n_lookback, target_token_size, tokenizer):
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
            input_coordinates_batch = input_coordinates.to(device)
            target_coordinates_batch = target_coordinates.to(device)
            
            prediction = model(X_batch, input_coordinates_batch, target_coordinates_batch)

            predictions.append(prediction.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    targets = targets.reshape((n_samples, n_lookback, n_channels))
    predictions = predictions.reshape(n_samples, target_token_size, n_channels)

    # Apply inverse tokenizer on predictions if applicable
    if tokenizer is not None:
        predictions = tokenizer.inverse_tokenize('target', predictions)

    return targets, predictions
