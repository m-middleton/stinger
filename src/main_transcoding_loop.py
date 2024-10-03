'''

'''
import sys
sys.path.insert(1, '../')

import os
import csv
import gc
import random
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

import multiprocessing as mp
from functools import partial

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import torch.nn as nn

import wandb

from processing.Format_Data import grab_ordered_windows, grab_random_windows, find_indices, EEGfNIRSData
from processing.FPCA import MultiChannelFPCA
from utilities.Read_Data import read_subjects_data
from models.Model_Utilities import predict_eeg, create_rnn, create_mlp, create_transformer
from utilities.utilities import calculate_channel_distances
from utilities.Plotting import plot_scatter_between_timepoints, compare_real_vs_predicted_erp, process_channel_rolling_correlation
from plotting.Windowed_Correlation import rolling_correlation

from config.Constants import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {DEVICE}')

# create directorys if they dont exist
if not os.path.exists(MODEL_WEIGHTS):
    os.makedirs(MODEL_WEIGHTS)
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

model_functions = {
    'rnn': create_rnn,
    'mlp': create_mlp,
    'transformer': create_transformer
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(
        subject_id, 
        model_config, 
        data_config, 
        training_config, 
        visualization_config
):
    # Convert all config values to strings
    def stringify_config(config):
        if isinstance(config, dict):
            return {k: stringify_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [stringify_config(item) for item in config]
        else:
            return str(config)

    model_config = stringify_config(model_config)
    data_config = stringify_config(data_config)
    training_config = stringify_config(training_config)
    visualization_config = stringify_config(visualization_config)

    # Create a dictionary containing all configurations
    config = {
        "subject_id": subject_id,
        "model_config": model_config,
        "data_config": data_config,
        "training_config": training_config,
        "visualization_config": visualization_config
    }
    
    # Get the output path from visualization_config
    output_path = visualization_config['output_path']
    
    # Create a filename for the config log
    config_filename = f"config_log_subject_{subject_id}.json"
    
    # Full path for the config log file
    config_file_path = os.path.join(output_path, config_filename)
    
    # Save the config dictionary as a JSON file
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to: {config_file_path}")

def get_data(subject_id_int, data_config, training_config):

    # Read and preprocess data
    eeg_raw_mne, nirs_raw_mne = read_subjects_data(
        subjects=[f'VP{subject_id_int:03d}'],
        raw_data_directory=RAW_DIRECTORY,
        tasks=data_config['tasks'],
        eeg_event_translations=EEG_EVENT_TRANSLATIONS,
        nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
        eeg_coords=EEG_COORDS,
        tasks_stimulous_to_crop=TASK_STIMULOUS_TO_CROP,
        trial_to_check_nirs=TRIAL_TO_CHECK_NIRS,
        eeg_t_min=data_config['eeg_t_min'],
        eeg_t_max=data_config['eeg_t_max'],
        nirs_t_min=data_config['nirs_t_min'],
        nirs_t_max=data_config['nirs_t_max'],
        eeg_sample_rate=data_config['eeg_sample_rate'],
        redo_preprocessing=False,
    )

    fnirs_sample_rate = nirs_raw_mne.info['sfreq']

    #remove HEOG and VEOG
    eeg_raw_mne.drop_channels(['HEOG', 'VEOG'])
    # get only hbo
    nirs_raw_mne.pick(picks='hbo')

    mrk_data, single_events_dict = mne.events_from_annotations(eeg_raw_mne)
    mrk_data[:,0] -= mrk_data[0,0]
    mrk_data[:,0] += (1*data_config['eeg_sample_rate'])
    print(single_events_dict)
    
    eeg_data = eeg_raw_mne.get_data()
    nirs_data = nirs_raw_mne.get_data()
    print(f'EEG Shape: {eeg_data.shape}') # n_channels x n_samples_eeg
    print(f'NIRS Shape: {nirs_data.shape}') # n_channels x n_samples_nirs
    print(f'MRK Shape: {mrk_data.shape}') # n_events x 3 (timestamp, event_type, event_id)

    # get channels
    eeg_data = eeg_data[data_config['eeg_channel_index']]
    nirs_data = nirs_data[data_config['nirs_channel_index']]

    # split train and test on eeg_data, nirs_data, and mrk_data
    eeg_test_size = int(eeg_data.shape[1]*training_config['test_size'])
    eeg_validation_size = int(eeg_data.shape[1]*training_config['validation_size'])
    eeg_train_size = eeg_data.shape[1] - eeg_test_size - eeg_validation_size

    nirs_test_size = int(nirs_data.shape[1]*training_config['test_size'])
    nirs_validation_size = int(nirs_data.shape[1]*training_config['validation_size'])
    nirs_train_size = nirs_data.shape[1] - nirs_test_size - nirs_validation_size

    train_eeg_data = eeg_data[:, :eeg_train_size]
    validation_eeg_data = eeg_data[:, eeg_train_size:eeg_train_size+eeg_validation_size]
    train_nirs_data = nirs_data[:, :nirs_train_size]
    validation_nirs_data = nirs_data[:, nirs_train_size:nirs_train_size+nirs_validation_size]

    test_eeg_data = eeg_data[:, eeg_train_size+eeg_validation_size:]
    test_nirs_data = nirs_data[:, nirs_train_size+nirs_validation_size:]

    # # normalize data
    train_eeg_data = (train_eeg_data - np.mean(train_eeg_data)) / np.std(train_eeg_data)
    validation_eeg_data = (validation_eeg_data - np.mean(validation_eeg_data)) / np.std(validation_eeg_data)
    test_eeg_data = (test_eeg_data - np.mean(test_eeg_data)) / np.std(test_eeg_data)
    train_nirs_data = (train_nirs_data - np.mean(train_nirs_data)) / np.std(train_nirs_data)
    validation_nirs_data = (validation_nirs_data - np.mean(validation_nirs_data)) / np.std(validation_nirs_data)
    test_nirs_data = (test_nirs_data - np.mean(test_nirs_data)) / np.std(test_nirs_data)

    print(f'Train EEG Shape: {train_eeg_data.shape}')
    print(f'Train NIRS Shape: {train_nirs_data.shape}')
    print(f'Validation EEG Shape: {validation_eeg_data.shape}')
    print(f'Validation NIRS Shape: {validation_nirs_data.shape}')
    print(f'Test EEG Shape: {test_eeg_data.shape}')
    print(f'Test NIRS Shape: {test_nirs_data.shape}')

    # Calculate train and test mrk_data
    train_max_event_timestamp = train_eeg_data.shape[1]
    validation_max_event_timestamp = validation_eeg_data.shape[1] + train_max_event_timestamp
    # MRK (timestamp, event_type, event_id)
    train_mrk_data = np.array([event for event in mrk_data if event[0] < train_max_event_timestamp])
    validation_mrk_data = np.array([event for event in mrk_data if event[0] >= train_max_event_timestamp and event[0] < validation_max_event_timestamp])
    test_mrk_data = np.array([event for event in mrk_data if event[0] >= validation_max_event_timestamp])
    # subtract train_max_event_timestamp from all index values in test_mrk_data
    validation_mrk_data[:,0] -= train_max_event_timestamp
    test_mrk_data[:,0] -= validation_max_event_timestamp

    print(train_mrk_data[:5])
    print(f'Train EEG Shape: {train_eeg_data.shape}')
    print(f'Train NIRS Shape: {train_nirs_data.shape}')
    print(f'Train MRK Shape: {train_mrk_data.shape}')
    print(f'Validation EEG Shape: {validation_eeg_data.shape}')
    print(f'Validation NIRS Shape: {validation_nirs_data.shape}')
    print(f'Validation MRK Shape: {validation_mrk_data.shape}')
    print(f'Test EEG Shape: {test_eeg_data.shape}')
    print(f'Test NIRS Shape: {test_nirs_data.shape}')
    print(f'Test MRK Shape: {test_mrk_data.shape}')

    # print counts of unique markers in train_mrk_data and test_mrk_data
    reverse_single_events_dict = {v: k for k, v in single_events_dict.items()}
    print("Train MRK Counts:")
    train_unique, train_counts = np.unique(train_mrk_data[:, 2], return_counts=True)
    for marker, count in zip(train_unique, train_counts):
        print(f"  Marker {reverse_single_events_dict[marker]}: {count}")

    print("\nValidation MRK Counts:")
    validation_unique, validation_counts = np.unique(validation_mrk_data[:, 2], return_counts=True)
    for marker, count in zip(validation_unique, validation_counts):
        print(f"  Marker {reverse_single_events_dict[marker]}: {count}")

    print("\nTest MRK Counts:")
    test_unique, test_counts = np.unique(test_mrk_data[:, 2], return_counts=True)
    for marker, count in zip(test_unique, test_counts):
        print(f"  Marker {reverse_single_events_dict[marker]}: {count}")

    # channel_names = ['Cz', 'Pz']  # Add more channel names as needed
    # event_selection = ['2-back non-target', '2-back target']
    # plot_erp_by_channel(train_eeg_data, train_mrk_data, single_events_dict, channel_names, event_selection)

    # grab coordinates at indexes
    eeg_coordinates = np.array(list(EEG_COORDS.values()))[data_config['eeg_channel_index']]
    nirs_coordinates = np.array(list(NIRS_COORDS.values()))[data_config['nirs_channel_index']]

    data_dict = {
        'train_eeg_data': train_eeg_data,
        'train_nirs_data': train_nirs_data,
        'test_eeg_data': test_eeg_data,
        'test_nirs_data': test_nirs_data,
        'train_mrk_data': train_mrk_data,
        'test_mrk_data': test_mrk_data,
        'validation_eeg_data': validation_eeg_data,
        'validation_nirs_data': validation_nirs_data,
        'validation_mrk_data': validation_mrk_data,

        'eeg_coordinates': eeg_coordinates,
        'nirs_coordinates': nirs_coordinates
    }

    return data_dict, fnirs_sample_rate, single_events_dict


def run_model(subject_id_int, model_name, model_config, data_config, training_config, visualization_config):
    subject_id = f'{subject_id_int:02d}'
    final_model_path = os.path.join(MODEL_WEIGHTS, f'{model_name}_{training_config["num_epochs"]}.pth')
    
    if visualization_config['do_wandb']:
        # Initialize wandb
        wandb.init(project="eeg-fnirs-transcoding",
               name=model_name,
               config={**model_config, **data_config, **training_config, **visualization_config})
    
    if os.path.exists(final_model_path) and not training_config['redo_train']:
        print(f'Model name exists, skipping {model_name}')
    else:
        print(f'Starting {model_name}')

        # Calculate spatial bias
        spatial_bias = torch.from_numpy(calculate_channel_distances(EEG_COORDS, NIRS_COORDS)).float()

        data_dict, fnirs_sample_rate, single_events_dict = get_data(subject_id_int, data_config, training_config)
        eeg_coordinates = torch.from_numpy(data_dict['eeg_coordinates']).float()
        nirs_coordinates = torch.from_numpy(data_dict['nirs_coordinates']).float()

        # Window data
        
        # Train data
        eeg_windowed_train, nirs_windowed_train, meta_data = grab_random_windows(
            nirs_data=data_dict['train_nirs_data'], 
            eeg_data=data_dict['train_eeg_data'],
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=data_config['eeg_sample_rate'],
            nirs_t_min=data_config['nirs_t_min'],
            nirs_t_max=data_config['nirs_t_max'],
            eeg_t_min=data_config['eeg_t_min'], 
            eeg_t_max=data_config['eeg_t_max'],
            number_of_windows=100)

        # Train data in order for visualization
        eeg_windowed_train_ordered, nirs_windowed_train_ordered, meta_data, train_mrk_data_ordered = grab_ordered_windows(
            nirs_data=data_dict['train_nirs_data'], 
            eeg_data=data_dict['train_eeg_data'],
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=data_config['eeg_sample_rate'],
            nirs_t_min=data_config['nirs_t_min'],
            nirs_t_max=data_config['nirs_t_max'],
            eeg_t_min=data_config['eeg_t_min'], 
            eeg_t_max=data_config['eeg_t_max'],
            markers=data_dict['train_mrk_data'])
        
        # validation data in order
        eeg_windowed_validation, nirs_windowed_validation, meta_data, validation_mrk_data_ordered = grab_ordered_windows(
            nirs_data=data_dict['validation_nirs_data'], 
            eeg_data=data_dict['validation_eeg_data'],
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=data_config['eeg_sample_rate'],
            nirs_t_min=data_config['nirs_t_min'],
            nirs_t_max=data_config['nirs_t_max'],
            eeg_t_min=data_config['eeg_t_min'], 
            eeg_t_max=data_config['eeg_t_max'],
            markers=data_dict['validation_mrk_data'])

        # Test data
        eeg_windowed_test, nirs_windowed_test, meta_data, test_mrk_data_ordered = grab_ordered_windows(
            nirs_data=data_dict['test_nirs_data'], 
            eeg_data=data_dict['test_eeg_data'],
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=data_config['eeg_sample_rate'],
            nirs_t_min=data_config['nirs_t_min'],
            nirs_t_max=data_config['nirs_t_max'],
            eeg_t_min=data_config['eeg_t_min'], 
            eeg_t_max=data_config['eeg_t_max'],
            markers=data_dict['test_mrk_data'])

        # Tokenization
        if data_config['tokenization'] == 'fpca':
            multi_channel_fpca = MultiChannelFPCA(subject_id=subject_id, 
                                    train_nirs_data=data_dict['train_nirs_data'], 
                                    train_eeg_data=data_dict['train_eeg_data'], 
                                    nirs_token_size=data_config['nirs_token_size'],
                                    eeg_token_size=data_config['eeg_token_size'],
                                    fnirs_sample_rate=fnirs_sample_rate,
                                    eeg_sample_rate=data_config['eeg_sample_rate'],
                                    nirs_channel_names=data_config['nirs_channels_to_use_base'],
                                    eeg_channel_names=data_config['eeg_channels_to_use'],
                                    model_weights=MODEL_WEIGHTS,
                                    nirs_t_min=data_config['nirs_t_min'],
                                    nirs_t_max=data_config['nirs_t_max'],
                                    eeg_t_min=data_config['eeg_t_min'],
                                    eeg_t_max=data_config['eeg_t_max'])

            # plot variance explained
            multi_channel_fpca.plot_explained_variance_over_dict(
                dict_name='nirs',
                path=os.path.join(visualization_config['output_path'], f'variance_nirs_fpca.jpeg'))
            multi_channel_fpca.plot_explained_variance_over_dict(
                dict_name='eeg',
                path=os.path.join(visualization_config['output_path'], f'variance_eeg_fpca.jpeg'))
            
            # plot pca shape tsne
            multi_channel_fpca.plot_shape(
                dict_name='nirs',
                path=os.path.join(visualization_config['output_path'], f'fpca_shape_tsne_nirs_fpca.jpeg'))
            multi_channel_fpca.plot_shape(
                dict_name='eeg',
                path=os.path.join(visualization_config['output_path'], f'fpca_shape_tsne_eeg_fpca.jpeg'))
            
            # Apply FPCA

            # Training
            nirs_windowed_train = multi_channel_fpca.perform_fpca_over_channels(dict_name='nirs',
                                                                                data=nirs_windowed_train,
                                                                                n_components=data_config['nirs_token_size'])
            eeg_windowed_train = multi_channel_fpca.perform_fpca_over_channels(dict_name='eeg',
                                                                                data=eeg_windowed_train, 
                                                                                n_components=data_config['eeg_token_size'])
            # Testing
            nirs_windowed_train_ordered = multi_channel_fpca.perform_fpca_over_channels(dict_name='nirs',
                                                                                        data=nirs_windowed_train_ordered,
                                                                                        n_components=data_config['nirs_token_size'])
            nirs_windowed_validation = multi_channel_fpca.perform_fpca_over_channels(dict_name='nirs',
                                                                                     data=nirs_windowed_validation,
                                                                                     n_components=data_config['nirs_token_size'])
            nirs_windowed_test = multi_channel_fpca.perform_fpca_over_channels(dict_name='nirs',
                                                                               data=nirs_windowed_test,
                                                                               n_components=data_config['nirs_token_size'])
            tokenizer = multi_channel_fpca
        elif data_config['tokenization'] == 'raw':
            # Use raw data without tokenization
            pass
        else:
            raise ValueError(f"Unknown tokenization method: {data_config['tokenization']}")

        n_channels = nirs_windowed_train.shape[1]

        # Append to the preallocated arrays
        eeg_windowed_train = eeg_windowed_train.transpose(0,2,1)
        nirs_windowed_train = nirs_windowed_train.transpose(0,2,1)
        
        eeg_windowed_train_ordered = eeg_windowed_train_ordered.transpose(0,2,1)
        nirs_windowed_train_ordered = nirs_windowed_train_ordered.transpose(0,2,1)

        eeg_windowed_validation = eeg_windowed_validation.transpose(0,2,1)
        nirs_windowed_validation = nirs_windowed_validation.transpose(0,2,1)

        eeg_windowed_test = eeg_windowed_test.transpose(0,2,1)
        nirs_windowed_test = nirs_windowed_test.transpose(0,2,1)

        print(f'Tokenized EEG Train Shape: {eeg_windowed_train.shape}')
        print(f'Tokenized NIRS Train Shape: {nirs_windowed_train.shape}')
        print(f'EEG Train Ordered Shape: {eeg_windowed_train_ordered.shape}')
        print(f'NIRS Train Ordered Shape: {nirs_windowed_train_ordered.shape}')

        print(f'EEG Test Shape: {eeg_windowed_test.shape}')
        print(f'NIRS Test Shape: {nirs_windowed_test.shape}')
        print(f'Tokenized EEG Test Shape: {eeg_windowed_test.shape}')
        print(f'Tokenized NIRS Test Shape: {nirs_windowed_test.shape}')

        # Flatten the data
        # nirs_windowed_validation = nirs_windowed_validation.reshape(-1, len(data_config['nirs_channel_index'])*data_config['nirs_token_size'])
        # nirs_windowed_test = nirs_windowed_test.reshape(-1, len(data_config['nirs_channel_index'])*data_config['nirs_token_size'])
        # nirs_windowed_train_ordered = nirs_windowed_train_ordered.reshape(-1, len(data_config['nirs_channel_index'])*data_config['nirs_token_size'])

        data_dict = {
            'eeg_windowed_train': eeg_windowed_train,
            'nirs_windowed_train': nirs_windowed_train,
            'eeg_windowed_train_ordered': eeg_windowed_train_ordered,
            'nirs_windowed_train_ordered': nirs_windowed_train_ordered,
            'eeg_windowed_validation': eeg_windowed_validation,
            'nirs_windowed_validation': nirs_windowed_validation,
            'eeg_windowed_test': eeg_windowed_test,
            'nirs_windowed_test': nirs_windowed_test,
            'validation_mrk_data_ordered': validation_mrk_data_ordered,
            'test_mrk_data_ordered': test_mrk_data_ordered,
            'train_mrk_data_ordered': train_mrk_data_ordered,

            'spatial_bias': spatial_bias,
            'tokenizer': tokenizer,
            'eeg_coordinates': eeg_coordinates,
            'nirs_coordinates': nirs_coordinates
        }

        # Model creation and training
        model_params = {k: v for k, v in model_config.items() if k != 'name'}
        model = model_functions[model_config['name']](
            n_input=len(data_config['nirs_channel_index']) * data_config['nirs_token_size'],
            n_output=len(data_config['eeg_channel_index']) * data_config['eeg_token_size'],
            nirs_sequence_length=data_config['nirs_token_size'],
            eeg_sequence_length=data_config['eeg_token_size'],
            num_nirs_channels=len(data_config['nirs_channel_index']),
            num_eeg_channels=len(data_config['eeg_channel_index']),
            **model_params
        )

        if training_config['do_train']:
        
            train_model(model, 
                        data_dict=data_dict,
                        model_name=model_name,
                        eeg_token_size=data_config['eeg_token_size'],
                        eeg_channels_to_use=data_config['eeg_channels_to_use'],
                        visualization_config=visualization_config,
                        config=training_config)

        # Evaluation and plotting
        evaluate_and_plot(model, 
                      tokenizer=tokenizer,
                      data_dict=data_dict,
                      single_events_dict=single_events_dict, 
                      model_name=model_name,
                      data_config=data_config,
                      visualization_config=visualization_config)
        
        if visualization_config['do_wandb']:
            wandb.finish()

def train_model(model, 
                data_dict,
                model_name,
                eeg_token_size,
                eeg_channels_to_use,
                visualization_config,
                config):
    model.to(DEVICE)

    nirs_train_tensor = data_dict['nirs_windowed_train']
    eeg_train_tensor = data_dict['eeg_windowed_train']
    
    nirs_train_tensor = torch.from_numpy(nirs_train_tensor).float()
    eeg_train_tensor = torch.from_numpy(eeg_train_tensor).float()
    # meta_data_tensor = torch.from_numpy(np.array(meta_data)).float()
    
    dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Perform inference on validation
    nirs_validation_tensor = torch.from_numpy(data_dict['nirs_windowed_validation']).float()
    eeg_validation_tensor = torch.from_numpy(data_dict['eeg_windowed_validation']).float()

    validation_dataset = EEGfNIRSData(nirs_validation_tensor, eeg_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    validation_dataset = EEGfNIRSData(nirs_validation_tensor, eeg_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    
    latest_epoch = 0
    loss_list = []
    if config['do_load']:
        model_path = f'{model_name}_epoch_1.pth'

        # find the latest model
        for file in os.listdir(MODEL_WEIGHTS):
            if file.startswith(f'{model_name}_epoch_'):
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    model_path = file
        print(f'Using Model Weights: {model_path}')
        model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
    
        # load loss list
        with open(os.path.join(MODEL_WEIGHTS, f'loss_{model_name}_{latest_epoch}.csv'), 'r') as file_ptr:
            reader = csv.reader(file_ptr)
            loss_list = list(reader)[0]
        print(f'Last loss: {float(loss_list[-1])/len(train_loader):.4f}')

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    class AmplitudeMSELoss(nn.Module):
        def __init__(self, amplitude_weight=0.5):
            super().__init__()
            self.amplitude_weight = amplitude_weight
            self.mse = nn.MSELoss()

        def forward(self, predictions, targets):
            mse_loss = self.mse(predictions, targets)
            
            # Calculate amplitude difference
            pred_amplitude = torch.max(predictions, dim=1)[0] - torch.min(predictions, dim=1)[0]
            target_amplitude = torch.max(targets, dim=1)[0] - torch.min(targets, dim=1)[0]
            amplitude_loss = self.mse(pred_amplitude, target_amplitude)
            
            return mse_loss + self.amplitude_weight * amplitude_loss

    loss_function = AmplitudeMSELoss(amplitude_weight=0.5)
    loss_function = torch.nn.MSELoss()

    train_validation_loss_dict = {'train':[], 'validation':[]}
    for epoch in range(latest_epoch, config['num_epochs']):
        model.train()
        total_loss = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            # spatial_bias_batch = data_dict['spatial_bias'].to(DEVICE)
            eeg_coordinates = data_dict['eeg_coordinates'].to(DEVICE).float()
            nirs_coordinates = data_dict['nirs_coordinates'].to(DEVICE).float()

            
            # Forward pass
            predictions = model(X_batch, nirs_coordinates, eeg_coordinates)
            # predictions = model(X_batch)
            predictions = predictions.reshape(y_batch.shape)

            # Loss calculation
            loss = loss_function(predictions, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        loss_list.append(total_loss)

        if visualization_config['do_wandb']:
            # Log to wandb
            wandb.log({"train_loss": total_loss / len(train_loader), "epoch": epoch + 1})

        if (epoch+1) % 50 == 0:
            # Save model weights
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS, f'{model_name}_{epoch+1}.pth'))
            with open(os.path.join(MODEL_WEIGHTS,f'loss_{model_name}_{epoch+1}.csv'), 'w', newline='') as file_ptr:
                wr = csv.writer(file_ptr, quoting=csv.QUOTE_ALL)
                wr.writerow(loss_list)

            targets, predictions = predict_eeg(model, 
                                                data_loader=validation_loader, 
                                                spatial_bias=data_dict['spatial_bias'],
                                                nirs_coordinates=data_dict['nirs_coordinates'],
                                                eeg_coordinates=data_dict['eeg_coordinates'],
                                                n_samples=data_dict['eeg_windowed_validation'].shape[0], 
                                                n_channels=data_dict['eeg_windowed_validation'].shape[2], 
                                                n_lookback=data_dict['eeg_windowed_validation'].shape[1],
                                                eeg_token_size=eeg_token_size,
                                                tokenizer=data_dict['tokenizer'])

            all_r2 = []
            for channel_id in range(len(eeg_channels_to_use)):
                targets_single = targets[:,:,channel_id]
                predictions_single = predictions[:,:,channel_id]

                target_average = np.mean(targets_single, axis=0)
                prediction_average = np.mean(predictions_single, axis=0)

                r2 = r2_score(target_average, prediction_average)
                all_r2.append(r2)
        
            total_r2 = np.max(all_r2)
            train_validation_loss_dict['train'].append(total_loss / len(train_loader))
            train_validation_loss_dict['validation'].append(total_r2)

            if visualization_config['do_wandb']:
                # Log to wandb
                wandb.log({
                    "validation_r2": total_r2,
                })

            print(f'Epoch: {epoch+1}, Train Loss: {total_loss / len(train_loader):.10f}, Validation R2: {total_r2:.10f}')
        elif (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Average Loss: {total_loss / len(train_loader):.10f}')

    return model

def evaluate_and_plot(model, 
                      tokenizer,
                      data_dict,
                      single_events_dict, 
                      model_name, 
                      data_config,
                      visualization_config):
    # Perform inference on ordered train
    nirs_train_tensor = torch.from_numpy(data_dict['nirs_windowed_train_ordered']).float()
    eeg_train_tensor = torch.from_numpy(data_dict['eeg_windowed_train_ordered']).float()

    train_dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Perform inference on ordered test
    nirs_test_tensor = torch.from_numpy(data_dict['nirs_windowed_test']).float()
    eeg_test_tensor = torch.from_numpy(data_dict['eeg_windowed_test']).float()

    test_dataset = EEGfNIRSData(nirs_test_tensor, eeg_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Channel IDX to use
    channels_to_use_dict = {channel_name:data_config['eeg_channels_to_use'].index(channel_name) for channel_name in visualization_config['channel_names']}
    
    # Get weights for specific epoch
    for weight_epoch in visualization_config['weight_epochs']:
        model_path = f'{model_name}_{weight_epoch}.pth'
        model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
        model.to(DEVICE)

        model.eval()

        targets_train, predictions_train = predict_eeg(model, 
                                            data_loader=train_loader, 
                                            spatial_bias=data_dict['spatial_bias'], 
                                            nirs_coordinates=data_dict['nirs_coordinates'],
                                            eeg_coordinates=data_dict['eeg_coordinates'],
                                            n_samples=data_dict['eeg_windowed_train_ordered'].shape[0], 
                                            n_channels=data_dict['eeg_windowed_train_ordered'].shape[2], 
                                            n_lookback=data_dict['eeg_windowed_train_ordered'].shape[1],
                                            eeg_token_size=data_config['eeg_token_size'],
                                            tokenizer=tokenizer)
        targets_test, predictions_test = predict_eeg(model, 
                                            data_loader=test_loader, 
                                            spatial_bias=data_dict['spatial_bias'], 
                                            nirs_coordinates=data_dict['nirs_coordinates'],
                                            eeg_coordinates=data_dict['eeg_coordinates'],
                                            n_samples=data_dict['eeg_windowed_test'].shape[0], 
                                            n_channels=data_dict['eeg_windowed_test'].shape[2], 
                                            n_lookback=data_dict['eeg_windowed_test'].shape[1],
                                            eeg_token_size=data_config['eeg_token_size'],
                                            tokenizer=tokenizer)
        
        channel_names = list(channels_to_use_dict.keys())
        channel_indexes = list(channels_to_use_dict.values())
        targets_train = targets_train[:,:,channel_indexes]
        predictions_train = predictions_train[:,:,channel_indexes]
        targets_test = targets_test[:,:,channel_indexes]
        predictions_test = predictions_test[:,:,channel_indexes]
        
        print(f'Weight Epoch: {weight_epoch}')
        print(f'Train Predictions Shape: {predictions_train.shape}')
        print(f'Train Targets Shape: {targets_train.shape}')
        print(f'Test Predictions Shape: {predictions_test.shape}')
        print(f'Test Targets Shape: {targets_test.shape}')
    
        # scipy.io.savemat(os.path.join(visualization_config['output_path'], f'test_{model_name}_{weight_epoch}.mat'), {'X': targets, 
        #                                                         'XPred':predictions,
        #                                                     'bins':10,
        #                                                     'scale':10,
        #                                       F              'srate':200})
        
        ''' Plotting target vs. output on concatenated data subplots for each channel '''

        offset = 0  # Assuming no offset
        timeSigma = 100  # Assuming a given timeSigma
        num_bins = 50  # Assuming a given number of bins

        highest_correlation = 0
        num_channels = len(channels_to_use_dict)
        fig, axs = plt.subplots(num_channels, 2, figsize=(18, 5 * num_channels), squeeze=False)
        print(f'Plotting target vs output for {num_channels} channels')

        # Pre-compute channel labels
        channel_labels = list(EEG_COORDS.keys())

        # Prepare data outside the loop
        train_data = (targets_train, predictions_train, 'Train', True, False)
        test_data = (targets_test, predictions_test, 'Test', False, True)

        for i, (channel_name, idx) in enumerate(channels_to_use_dict.items()):
            print(f'Channel Name: {channel_name} Remaining: {len(channels_to_use_dict)-i}')
            for j, (targets, predictions, type_label, do_legend, do_colorbar) in enumerate([train_data, test_data]):
                targets_single = targets[:,:,i].reshape(1, -1)
                predictions_single = predictions[:,:,i].reshape(1, -1)

                # r2 = r2_score(targets_single, predictions_single)
                # if j == 1 and r2 > highest_r2:
                #     highest_r2 = r2

                _, average_correlation = rolling_correlation(targets_single, 
                                    predictions_single, 
                                    [channel_name], 
                                    offset=offset,
                                    sampling_frequency=data_config['eeg_sample_rate'],
                                    timeSigma=timeSigma, 
                                    num_bins=num_bins, 
                                    zoom_start=0, 
                                    do_legend=do_legend,
                                    do_colorbar=do_colorbar,
                                    ax=axs[i, j], 
                                    title='')
                
                if j==1 and average_correlation > highest_correlation:
                    highest_correlation = average_correlation
                
                axs[i, j].text(0.5, 0.9, f'{channel_name} {type_label} Cor: {average_correlation:.10f}', 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=axs[i, j].transAxes)

        plt.tight_layout()
        fig.savefig(os.path.join(visualization_config['output_path'], f'summary_correlation_{highest_correlation:.10f}_{model_name}_{weight_epoch}.jpeg'))
        plt.close()

        ''' Compare train and test predictions and ERPS'''
        # concatenate windows targets and predictions (sample, window, channel) -> (sample*window, channel)
        targets_train = targets_train.reshape(-1, len(channels_to_use_dict))
        predictions_train = predictions_train.reshape(-1, len(channels_to_use_dict))

        targets_test = targets_test.reshape(-1, len(channels_to_use_dict))
        predictions_test = predictions_test.reshape(-1, len(channels_to_use_dict))

        # plot scatter between timepoints in predicted vs real for both train and test
        fig, axs = plot_scatter_between_timepoints(
            targets_train,
            predictions_train,
            targets_test,
            predictions_test,
            channel_names
        )
        
        fig.savefig(os.path.join(visualization_config['output_path'], f'{highest_correlation:.10f}_{model_name}_{weight_epoch}_scatter_combined.jpeg'))
        plt.close()

        # Train predicted vs real ERP from train_mrk
        print('Train ERP')
        fig, axs = compare_real_vs_predicted_erp(targets_train.T, 
                                        predictions_train.T, 
                                        data_dict['train_mrk_data_ordered'], 
                                        single_events_dict, 
                                        channel_names, 
                                        visualization_config['events'])
        
        fig.savefig(os.path.join(visualization_config['output_path'], f'train_{highest_correlation:.10f}_{model_name}_{weight_epoch}_erp.jpeg'))
        plt.close()

        # Test predicted vs real ERP from test_mrk
        print('Test ERP')
        fig, axs = compare_real_vs_predicted_erp(targets_test.T, 
                                        predictions_test.T, 
                                        data_dict['test_mrk_data_ordered'], 
                                        single_events_dict, 
                                        channel_names, 
                                        visualization_config['events'])
        
        fig.savefig(os.path.join(visualization_config['output_path'], f'test_{highest_correlation:.10f}_{model_name}_{weight_epoch}_erp.jpeg'))
        plt.close()

        if visualization_config['do_wandb']:
            # Log final metrics to wandb
            wandb.log({
                "final_test_correlation": highest_correlation,
                "weight_epoch": weight_epoch
            })

def main():
    set_seed(42)  # You can choose any integer as your seed

    # get_hrf(subject_ids)
    # input('Press Enter to continue...')

    # plot_grand_average()
    # plot_erp_matrix(subject_ids=subject_ids,
    #                 nirs_test_channels=list(NIRS_COORDS.keys()),
    #                 eeg_test_channels=EEG_CHANNEL_NAMES,
    #                 test_events=['2-back non-target', '2-back target'])
    

    # Make a new timestamp folder at OUTPUT_DIRECTORY for this run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_output_directory = os.path.join(OUTPUT_DIRECTORY, timestamp)
    os.makedirs(base_output_directory, exist_ok=True)

    # Define channels to use
    nirs_channels_to_use_base = list(NIRS_COORDS.keys())
    nirs_channel_index = find_indices(list(NIRS_COORDS.keys()),nirs_channels_to_use_base)

    eeg_channels_to_use = list(EEG_COORDS.keys())
    eeg_channel_index = find_indices(EEG_CHANNEL_NAMES,eeg_channels_to_use)

    # Configuration
    data_config = {
        'tasks': ['nback'],
        'nirs_channels_to_use_base': nirs_channels_to_use_base,
        'eeg_channels_to_use': eeg_channels_to_use,
        'eeg_channel_index': eeg_channel_index,
        'nirs_channel_index': nirs_channel_index,
        'eeg_sample_rate': 200,
        'eeg_t_min': 0,
        'eeg_t_max': 1,
        'nirs_t_min': -5,
        'nirs_t_max': 10,
        # 'nirs_token_size': 20,
        # 'eeg_token_size': 10,
        'tokenization': 'fpca',  # or 'raw'
    }

    visualization_config = {
        'do_wandb': False,
        'do_plot': True,
        'do_save': True,
        'weight_epochs': [500, 1000],
        'channel_names': list(EEG_COORDS.keys()),
        'events': ['2-back non-target', '2-back target', '3-back non-target', '3-back target'],
        'output_path': base_output_directory
    }

    model_configs = [
        {
        'name': 'rnn',
        'hidden_dim': 64,
        }
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
        }
    ]

    token_sizes = [
        {
            'nirs_token_size': 20,
            'eeg_token_size': 10,
        },
    ]

    # from config.training_dictionaries import model_configs, training_configs


    run_number = 1
    subject_ids = [1] #range(4, 5)  # 1-27
    for subject_id in subject_ids:
        for token_size in token_sizes:
            for model_config in model_configs:
                for training_config in training_configs:
                    model_name = f'{model_config["name"]}_{subject_id:02d}_n_chs_{len(data_config["nirs_channel_index"])}_{len(data_config["eeg_channel_index"])}_{run_number}'
                    # Timestamp for loop
                    output_directory = os.path.join(base_output_directory, model_name)
                    os.makedirs(output_directory, exist_ok=True)

                    print(f"Running model for subject {subject_id} with config:")
                    print(f"Model config: {model_config}")
                    print(f"Training config: {training_config}")
                    print(f"Token size: {token_size}")

                    data_config = data_config | token_size
                    visualization_config['output_path'] = output_directory
                    save_config(subject_id, model_config, data_config, training_config, visualization_config)
                    
                    run_model(subject_id, model_name, model_config, data_config, training_config, visualization_config)
                    gc.collect()
                    torch.cuda.empty_cache()

                    run_number += 1

if __name__ == '__main__':
    main()

