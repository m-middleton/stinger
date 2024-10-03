token_sizes = [
    # {
    #     'nirs_token_size': 20,
    #     'eeg_token_size': 10,
    # },
    # {
    #     'nirs_token_size': 30,
    #     'eeg_token_size': 10,
    # },

    # {
    #     'nirs_token_size': 20,
    #     'eeg_token_size': 5,
    # },
    # {
    #     'nirs_token_size': 10,
    #     'eeg_token_size': 5,
    # },
    # {
    #     'nirs_token_size': 30,
    #     'eeg_token_size': 5,
    # },

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

# model_configs = [
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.2,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.3,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.1,
#         'activation': nn.ReLU,
#     },
    
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.2,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.3,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.1,
#         'activation': nn.ReLU,
#     },

#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.2,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.3,
#         'activation': nn.ReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.1,
#         'activation': nn.ReLU,
#     },

    
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.2,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.3,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [512, 256, 128],
#         'dropout_rate': 0.1,
#         'activation': nn.LeakyReLU,
#     },
    
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.2,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.3,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [1024, 512, 256],
#         'dropout_rate': 0.1,
#         'activation': nn.LeakyReLU,
#     },

#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.2,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.3,
#         'activation': nn.LeakyReLU,
#     },
#     {
#         'name': 'mlp',
#         'hidden_dims': [256, 128, 64],
#         'dropout_rate': 0.1,
#         'activation': nn.LeakyReLU,
#     },
# ]

model_configs = [
    {
        'name': 'rnn',
        'hidden_dim': 64,
    },
    {
        'name': 'rnn',
        'hidden_dim': 128,
    },
    {
        'name': 'rnn',
        'hidden_dim': 256,
    },
]

training_configs = [
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.001,
    },
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.0005,
    },
    {
        'do_train': True,
        'do_load': False,
        'redo_train': True,
        'num_epochs': 1000,
        'test_size': 0.15,
        'validation_size': 0.05,
        'learning_rate': 0.00005,
    },
    # {
    #     'do_train': True,
    #     'do_load': False,
    #     'redo_train': True,
    #     'num_epochs': 1000,
    #     'test_size': 0.15,
    #     'validation_size': 0.05,
    #     'learning_rate': 0.1,
    # },
]