import torch.nn as nn


training_configs = [
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'num_train_windows': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.001,
    },
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'num_train_windows': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.0005,
    },
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'num_train_windows': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.00005,
    },
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'num_train_windows': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.1,
    },
]

token_sizes = [
    {
        'nirs_token_size': 20,
        'eeg_token_size': 10,
    },
    {
        'nirs_token_size': 30,
        'eeg_token_size': 10,
    },

    {
        'nirs_token_size': 20,
        'eeg_token_size': 5,
    },
    {
        'nirs_token_size': 10,
        'eeg_token_size': 5,
    },
    {
        'nirs_token_size': 30,
        'eeg_token_size': 5,
    },

    {
        'nirs_token_size': 30,
        'eeg_token_size': 20,
    },
    {
        'nirs_token_size': 60,
        'eeg_token_size': 50,
    },
    {
        'nirs_token_size': 110,
        'eeg_token_size': 100,
    },
]

mlp_model_configs = [
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.2,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.1,
        'activation': nn.ReLU,
    },
    
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.2,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.3,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.1,
        'activation': nn.ReLU,
    },

    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.2,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'activation': nn.ReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.1,
        'activation': nn.ReLU,
    },

    
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.2,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.1,
        'activation': nn.LeakyReLU,
    },
    
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.2,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.3,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [1024, 512, 256],
        'dropout_rate': 0.1,
        'activation': nn.LeakyReLU,
    },

    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.2,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'activation': nn.LeakyReLU,
    },
    {
        'name': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.1,
        'activation': nn.LeakyReLU,
    },
]

rnn_model_configs = [
    # Baseline configuration
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0,
        'use_attention': False,
    },
    # Vary hidden dimension
    {
        'name': 'rnn',
        'hidden_dim': 128,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0,
        'use_attention': False,
    },
    # Vary number of LSTM layers
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 2,
        'bidirectional': False,
        'dropout': 0,
        'use_attention': False,
    },
    # Enable bidirectional LSTM
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': True,
        'dropout': 0,
        'use_attention': False,
    },
    # Add dropout
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0.2,
        'use_attention': False,
    },
    # Enable attention mechanism
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0,
        'use_attention': True,
        'attention_heads': 1,
    },
    # Disable spatial encoding
    {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': False,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0,
        'use_attention': False,
    },
    # Combine multiple changes (deeper network with attention)
    {
        'name': 'rnn',
        'hidden_dim': 128,
        'spatial_encoding': True,
        'num_lstm_layers': 2,
        'bidirectional': True,
        'dropout': 0.1,
        'use_attention': True,
        'attention_heads': 2,
    },
]
