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
import io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import concurrent.futures
from functools import partial

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import concurrent.futures

import wandb

from processing.Format_Data import grab_ordered_windows, grab_random_windows, SignalData  # Updated
from tokenization.FPCA import MultiChannelFPCA
from tokenization.Wavelet import MultiChannelWavelet
from tokenization.Raw_Tokenization import RawTokenizer
from utilities.Read_Data import get_data_nirs_eeg, get_data_eeg_to_eeg
from models.Model_Utilities import predict_signal, create_rnn, create_mlp, create_transformer  # Updated
from utilities.utilities import calculate_channel_distances
from utilities.Plotting import plot_scatter_between_timepoints, compare_real_vs_predicted_erp
from plotting.Windowed_Correlation import rolling_correlation, process_channel_rolling_correlation

from config.Constants import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {DEVICE}')

# create directories if they don't exist
if not os.path.exists(MODEL_WEIGHTS):
    os.makedirs(MODEL_WEIGHTS)
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

model_functions = {
    'rnn': create_rnn,
    'mlp': create_mlp,
    'transformer': create_transformer
}
data_functions = {
    'get_data_nirs_eeg': get_data_nirs_eeg,
    'get_data_eeg_to_eeg': get_data_eeg_to_eeg
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

def run_model(subject_id_int, model_name, model_config, data_config, training_config, visualization_config):
    subject_id = f'{subject_id_int:02d}'
    final_model_path = os.path.join(MODEL_WEIGHTS, f'{model_name}_{training_config["num_epochs"]}.pth')
    
    if visualization_config['do_wandb']:
        # Initialize wandb
        wandb.init(project="signal-transcoding",
                   name=model_name,
                   config={**model_config, **data_config, **training_config, **visualization_config})
    
    if os.path.exists(final_model_path) and not training_config['redo_train']:
        print(f'Model name exists, skipping {model_name}')
    else:
        print(f'Starting {model_name}')

        data_dict, input_sample_rate, target_sample_rate, single_events_dict = data_functions[data_config['get_data_function']](
            subject_id_int, 
            data_config, 
            training_config)
        
        input_coordinates = torch.from_numpy(data_dict['input_coordinates']).float()
        target_coordinates = torch.from_numpy(data_dict['target_coordinates']).float()

        # Calculate spatial bias (if needed)
        spatial_bias = torch.from_numpy(calculate_channel_distances(data_dict['input_coordinates'], data_dict['target_coordinates'])).float()

        # Model creation and training
        model_params = {k: v for k, v in model_config.items() if k != 'name'}
        model = model_functions[model_config['name']](
            n_input=len(data_config['input_channel_names']) * data_config['input_token_size'],
            n_output=len(data_config['target_channel_names']) * data_config['target_token_size'],
            input_sequence_length=data_config['input_token_size'],
            target_sequence_length=data_config['target_token_size'],
            num_input_channels=len(data_config['input_channel_names']),
            num_target_channels=len(data_config['target_channel_names']),
            **model_params
        )

        # Window data
        
        # Train data
        input_windowed_train, target_windowed_train, meta_data = grab_random_windows(
            input_signal=data_dict['train_input_signal'], 
            target_signal=data_dict['train_target_signal'],
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_t_min=data_config['input_t_min'],
            input_t_max=data_config['input_t_max'],
            target_t_min=data_config['target_t_min'],
            target_t_max=data_config['target_t_max'],
            number_of_windows=training_config['num_train_windows'])

        # Train data in order for visualization
        input_windowed_train_ordered, target_windowed_train_ordered, meta_data, train_event_data_ordered = grab_ordered_windows(
            input_signal=data_dict['train_input_signal'], 
            target_signal=data_dict['train_target_signal'],
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_t_min=data_config['input_t_min'],
            input_t_max=data_config['input_t_max'],
            target_t_min=data_config['target_t_min'],
            target_t_max=data_config['target_t_max'],
            events=data_dict['train_mrk_data'])
        
        # Validation data in order
        input_windowed_validation, target_windowed_validation, meta_data, validation_event_data_ordered = grab_ordered_windows(
            input_signal=data_dict['validation_input_signal'], 
            target_signal=data_dict['validation_target_signal'],
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_t_min=data_config['input_t_min'],
            input_t_max=data_config['input_t_max'],
            target_t_min=data_config['target_t_min'],
            target_t_max=data_config['target_t_max'],
            events=data_dict['validation_mrk_data'])

        # Test data
        input_windowed_test, target_windowed_test, meta_data, test_event_data_ordered = grab_ordered_windows(
            input_signal=data_dict['test_input_signal'], 
            target_signal=data_dict['test_target_signal'],
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_t_min=data_config['input_t_min'],
            input_t_max=data_config['input_t_max'],
            target_t_min=data_config['target_t_min'],
            target_t_max=data_config['target_t_max'],
            events=data_dict['test_mrk_data'])

        # Tokenization
        if data_config['tokenization'] == 'raw':
            # Use raw data without tokenization
            # uses a fake tokenizer
            tokenizer = RawTokenizer(
                subject_id=subject_id, 
                model_weights=MODEL_WEIGHTS,
                redo_tokenization=data_config['redo_tokenization']
            )
            
        elif data_config['tokenization'] == 'fpca':
            tokenizer = MultiChannelFPCA(
                subject_id=subject_id, 
                model_weights=MODEL_WEIGHTS,
                redo_tokenization=data_config['redo_tokenization']
            )
        elif data_config['tokenization'] == 'wavelets':
            tokenizer = MultiChannelWavelet(
                subject_id=subject_id, 
                model_weights=MODEL_WEIGHTS,
                redo_tokenization=data_config['redo_tokenization']
            )
        else:
            raise ValueError(f"Unknown tokenization method: {data_config['tokenization']}")
        
        # Tokenization Steps
        tokenizer.fit(
            train_input_signal=data_dict['train_input_signal'], 
            train_target_signal=data_dict['train_target_signal'], 
            input_token_size=data_config['input_token_size'],
            target_token_size=data_config['target_token_size'],
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_channel_names=data_config['input_channel_names'],
            target_channel_names=data_config['target_channel_names'],
            input_t_min=data_config['input_t_min'],
            input_t_max=data_config['input_t_max'],
            target_t_min=data_config['target_t_min'],
            target_t_max=data_config['target_t_max']
        )

        # Tokenization Training
        input_windowed_train = tokenizer.tokenize(data=input_windowed_train,signal_type='input')
        target_windowed_train = tokenizer.tokenize(data=target_windowed_train,signal_type='target')

        # Tokenization Testing
        input_windowed_train_ordered = tokenizer.tokenize(data=input_windowed_train_ordered,signal_type='input')
        input_windowed_validation = tokenizer.tokenize(data=input_windowed_validation,signal_type='input')
        input_windowed_test = tokenizer.tokenize(data=input_windowed_test,signal_type='input')

        n_input_channels = input_windowed_train.shape[1]
        n_target_channels = target_windowed_train.shape[1]

        # Transpose data to match expected input shape
        target_windowed_train = target_windowed_train.transpose(0, 2, 1)
        input_windowed_train = input_windowed_train.transpose(0, 2, 1)
        
        target_windowed_train_ordered = target_windowed_train_ordered.transpose(0, 2, 1)
        input_windowed_train_ordered = input_windowed_train_ordered.transpose(0, 2, 1)

        target_windowed_validation = target_windowed_validation.transpose(0, 2, 1)
        input_windowed_validation = input_windowed_validation.transpose(0, 2, 1)

        target_windowed_test = target_windowed_test.transpose(0, 2, 1)
        input_windowed_test = input_windowed_test.transpose(0, 2, 1)

        print(f'Tokenized Target Train Shape: {target_windowed_train.shape}')
        print(f'Tokenized Input Train Shape: {input_windowed_train.shape}')

        print(f'Train Target Ordered Shape: {target_windowed_train_ordered.shape}')
        print(f'Train Input Ordered Shape: {input_windowed_train_ordered.shape}')

        print(f'Validation Target Shape: {target_windowed_validation.shape}')
        print(f'Validation Input Tokenized Shape: {input_windowed_validation.shape}')

        print(f'Test Target Shape: {target_windowed_test.shape}')
        print(f'Test Input Tokenized Shape: {input_windowed_test.shape}')

        data_dict = {
            'target_windowed_train': target_windowed_train,
            'input_windowed_train': input_windowed_train,
            'target_windowed_train_ordered': target_windowed_train_ordered,
            'input_windowed_train_ordered': input_windowed_train_ordered,
            'target_windowed_validation': target_windowed_validation,
            'input_windowed_validation': input_windowed_validation,
            'target_windowed_test': target_windowed_test,
            'input_windowed_test': input_windowed_test,
            'validation_event_data_ordered': validation_event_data_ordered,
            'test_event_data_ordered': test_event_data_ordered,
            'train_event_data_ordered': train_event_data_ordered,
            'spatial_bias': spatial_bias,
            'tokenizer': tokenizer,
            'input_coordinates': input_coordinates,
            'target_coordinates': target_coordinates
        }

        if training_config['do_train']:
            train_model(model, 
                        data_dict=data_dict,
                        model_name=model_name,
                        input_token_size=data_config['input_token_size'],
                        target_token_size=data_config['target_token_size'],
                        input_channel_names=data_config['input_channel_names'],
                        target_channel_names=data_config['target_channel_names'],
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
                input_token_size,
                target_token_size,
                input_channel_names,
                target_channel_names,
                visualization_config,
                config):
    model.to(DEVICE)

    input_train_tensor = data_dict['input_windowed_train']
    target_train_tensor = data_dict['target_windowed_train']
    
    input_train_tensor = torch.from_numpy(input_train_tensor).float()
    target_train_tensor = torch.from_numpy(target_train_tensor).float()
    
    dataset = SignalData(input_train_tensor, target_train_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Validation data
    input_validation_tensor = torch.from_numpy(data_dict['input_windowed_validation']).float()
    target_validation_tensor = torch.from_numpy(data_dict['target_windowed_validation']).float()

    validation_dataset = SignalData(input_validation_tensor, target_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    
    latest_epoch = 0
    loss_list = []
    if config['do_load']:
        model_path = f'{model_name}_epoch_1.pth'

        # Find the latest model
        for file in os.listdir(MODEL_WEIGHTS):
            if file.startswith(f'{model_name}_epoch_'):
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    model_path = file
        print(f'Using Model Weights: {model_path}')
        model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
    
        # Load loss list
        with open(os.path.join(MODEL_WEIGHTS, f'loss_{model_name}_{latest_epoch}.csv'), 'r') as file_ptr:
            reader = csv.reader(file_ptr)
            loss_list = list(reader)[0]
        print(f'Last loss: {float(loss_list[-1])/len(train_loader):.4f}')

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    loss_function = torch.nn.MSELoss()

    train_validation_loss_dict = {'train':[], 'validation':[]}
    for epoch in range(latest_epoch, config['num_epochs']):
        model.train()
        total_loss = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            input_signal_coordinates = data_dict['input_coordinates'].to(DEVICE).float()
            target_signal_coordinates = data_dict['target_coordinates'].to(DEVICE).float()

            # Forward pass
            predictions = model(X_batch, input_signal_coordinates, target_signal_coordinates)
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

            targets, predictions = predict_signal(model, 
                                                  data_loader=validation_loader, 
                                                  spatial_bias=data_dict['spatial_bias'],
                                                  input_coordinates=data_dict['input_coordinates'],
                                                  target_coordinates=data_dict['target_coordinates'],
                                                  n_samples=data_dict['target_windowed_validation'].shape[0], 
                                                  n_channels=data_dict['target_windowed_validation'].shape[2], 
                                                  n_lookback=data_dict['target_windowed_validation'].shape[1],
                                                  target_token_size=target_token_size,
                                                  tokenizer=data_dict['tokenizer'])

            all_r2 = []
            for channel_id in range(len(target_channel_names)):
                targets_single = targets[:, :, channel_id]
                predictions_single = predictions[:, :, channel_id]

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

def process_single_channel_data(i, channel_name, targets_train, predictions_train, targets_test, predictions_test, 
                                targets_train_flat, predictions_train_flat, targets_test_flat, predictions_test_flat, 
                                data_dict, single_events_dict, visualization_config, data_config):
    
    results = {}
    results['channel_name'] = channel_name
    results['i'] = i
    
    # Prepare data for rolling correlation plots
    results['rolling_corr_data'] = []
    for targets, predictions, type_label in [
        (targets_train, predictions_train, 'Train'),
        (targets_test, predictions_test, 'Test')
    ]:
        targets_single = targets[:,:,i].reshape(1, -1)
        predictions_single = predictions[:,:,i].reshape(1, -1)
        results['rolling_corr_data'].append((targets_single, predictions_single, type_label))

    # Prepare data for scatter plots
    results['scatter_data'] = {
        'train': (targets_train_flat[:, i].reshape(-1, 1), predictions_train_flat[:, i].reshape(-1, 1)),
        'test': (targets_test_flat[:, i].reshape(-1, 1), predictions_test_flat[:, i].reshape(-1, 1))
    }

    # Prepare data for ERP plots
    results['erp_data'] = []
    for targets, predictions, events, label in [
        (targets_train_flat[:, i].reshape(1, -1), predictions_train_flat[:, i].reshape(1, -1), data_dict['train_event_data_ordered'], 'Train'),
        (targets_test_flat[:, i].reshape(1, -1), predictions_test_flat[:, i].reshape(1, -1), data_dict['test_event_data_ordered'], 'Test')
    ]:
        results['erp_data'].append((targets, predictions, events, label))

    print(f'Finished processing data for Channel: {channel_name} ({i+1}/{len(data_config["target_channel_names"])})')
    return results

def evaluate_and_plot(model, 
                      tokenizer,
                      data_dict,
                      single_events_dict,
                      model_name, 
                      data_config,
                      visualization_config):
    # Perform inference on ordered train
    input_train_tensor = torch.from_numpy(data_dict['input_windowed_train_ordered']).float()
    target_train_tensor = torch.from_numpy(data_dict['target_windowed_train_ordered']).float()

    train_dataset = SignalData(input_train_tensor, target_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Perform inference on ordered test
    input_test_tensor = torch.from_numpy(data_dict['input_windowed_test']).float()
    target_test_tensor = torch.from_numpy(data_dict['target_windowed_test']).float()

    test_dataset = SignalData(input_test_tensor, target_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Channel indexes to use
    channels_to_use_dict = {channel_name: data_config['target_channel_names'].index(channel_name) for channel_name in visualization_config['channel_names']}
    
    # Get weights for specific epoch
    for weight_epoch in visualization_config['weight_epochs']:
        model_path = f'{model_name}_{weight_epoch}.pth'
        model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
        model.to(DEVICE)

        model.eval()

        targets_train, predictions_train = predict_signal(model, 
                                            data_loader=train_loader, 
                                            spatial_bias=data_dict['spatial_bias'], 
                                            input_coordinates=data_dict['input_coordinates'],
                                            target_coordinates=data_dict['target_coordinates'],
                                            n_samples=data_dict['target_windowed_train_ordered'].shape[0], 
                                            n_channels=data_dict['target_windowed_train_ordered'].shape[2], 
                                            n_lookback=data_dict['target_windowed_train_ordered'].shape[1],
                                            target_token_size=data_config['target_token_size'],
                                            tokenizer=tokenizer)
        targets_test, predictions_test = predict_signal(model, 
                                            data_loader=test_loader, 
                                            spatial_bias=data_dict['spatial_bias'], 
                                            input_coordinates=data_dict['input_coordinates'],
                                            target_coordinates=data_dict['target_coordinates'],
                                            n_samples=data_dict['target_windowed_test'].shape[0], 
                                            n_channels=data_dict['target_windowed_test'].shape[2], 
                                            n_lookback=data_dict['target_windowed_test'].shape[1],
                                            target_token_size=data_config['target_token_size'],
                                            tokenizer=tokenizer)
        
        channel_names = list(channels_to_use_dict.keys())
        channel_indexes = list(channels_to_use_dict.values())
        targets_train = targets_train[:, :, channel_indexes]
        predictions_train = predictions_train[:, :, channel_indexes]
        targets_test = targets_test[:, :, channel_indexes]
        predictions_test = predictions_test[:, :, channel_indexes]
        
        print(f'Weight Epoch: {weight_epoch}')
        print(f'Train Predictions Shape: {predictions_train.shape}')
        print(f'Train Targets Shape: {targets_train.shape}')
        print(f'Test Predictions Shape: {predictions_test.shape}')
        print(f'Test Targets Shape: {targets_test.shape}')
    
        # scipy.io.savemat(os.path.join(visualization_config['output_path'], f'test_{model_name}_{weight_epoch}.mat'), {'X': targets, 
        #                                                         'XPred':predictions,
        #                                                     'bins':10,
        #                                                     'scale':10,
        #                                                     'srate':200})
        
        ''' Combine all evaluation plots into a single figure '''
        num_channels = len(channels_to_use_dict)
        num_plot_types = 5  # Rolling correlation (train/test), Scatter (2 subplots), ERP (1 subplot)
        fig, axs = plt.subplots(num_channels, num_plot_types, figsize=(30, 10*num_channels), squeeze=False)
        
        highest_correlation = 0
        print(f'Plotting combined evaluation for {num_channels} channels')

        # Pre-compute channel labels
        channel_labels = list(channels_to_use_dict.keys())

        # Flatten the data for ERP and scatter plots
        targets_train_flat = targets_train.reshape(targets_train.shape[0] * targets_train.shape[1], -1)
        predictions_train_flat = predictions_train.reshape(predictions_train.shape[0] * predictions_train.shape[1], -1)
        targets_test_flat = targets_test.reshape(targets_test.shape[0] * targets_test.shape[1], -1)
        predictions_test_flat = predictions_test.reshape(predictions_test.shape[0] * predictions_test.shape[1], -1)

        num_channels = len(channels_to_use_dict)
        num_plot_types = 5
        fig, axs = plt.subplots(num_channels, num_plot_types, figsize=(30, 10*num_channels), squeeze=False)

        highest_correlation = 0
        print(f'Processing rolling correlations for {num_channels} channels')

        # Pre-compute channel labels
        channel_labels = list(channels_to_use_dict.keys())

        # Flatten the data for scatter plots
        targets_train_flat = targets_train.reshape(targets_train.shape[0] * targets_train.shape[1], -1)
        predictions_train_flat = predictions_train.reshape(predictions_train.shape[0] * predictions_train.shape[1], -1)
        targets_test_flat = targets_test.reshape(targets_test.shape[0] * targets_test.shape[1], -1)
        predictions_test_flat = predictions_test.reshape(predictions_test.shape[0] * predictions_test.shape[1], -1)

        # Prepare the partial function for multiprocessing
        process_func = partial(
            process_channel_rolling_correlation,
            targets_train=targets_train,
            predictions_train=predictions_train,
            targets_test=targets_test,
            predictions_test=predictions_test,
            data_config=data_config
        )

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_channel = {executor.submit(process_func, channel_name, i): (channel_name, i) 
                                for i, channel_name in enumerate(channel_labels)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_channel):
                channel_name, i, results = future.result()
                print(f'Finished processing for Channel: {channel_name} ({i+1}/{num_channels})')
                
                # Plot the results
                for j, (type_label, average_correlation, correlation_fig) in enumerate(results):
                    # Copy the figure to the main plot
                    correlation_ax = correlation_fig.axes[0]
                    axs[i, j].clear()
                    axs[i, j].imshow(correlation_ax.images[0].get_array(), aspect='auto', 
                                    extent=correlation_ax.images[0].get_extent(),
                                    cmap=correlation_ax.images[0].get_cmap())
                    axs[i, j].set_xlabel(correlation_ax.get_xlabel())
                    axs[i, j].set_ylabel(correlation_ax.get_ylabel())
                    
                    if type_label == 'Test' and average_correlation > highest_correlation:
                        highest_correlation = average_correlation
                    
                    axs[i, j].set_title(f'{channel_name} {type_label}\nCorr: {average_correlation:.4f}')
                    plt.close(correlation_fig)  # Close the individual figure to free up memory

            # Scatter plot
            plot_scatter_between_timepoints(
                targets_train_flat,
                predictions_train_flat,
                targets_test_flat,
                predictions_test_flat,
                [channel_name],
                ax=axs[i, 2:4]
            )

            # # ERP plots
            # number_of_events = len(visualization_config['events'])
            # # ERP plots
            # erp_axs = fig.add_subplot(num_channels, num_plot_types, (i*num_plot_types)+4)
            # for j, (targets, predictions, events, label) in enumerate([
            #     (targets_train_flat.T, predictions_train_flat.T, data_dict['train_event_data_ordered'], 'Train'),
            #     (targets_test_flat.T, predictions_test_flat.T, data_dict['test_event_data_ordered'], 'Test')
            # ]):
            #     ax_number = 4 if j == 0 else 4 + number_of_events
            #     compare_real_vs_predicted_erp(targets, 
            #                                 predictions, 
            #                                 events, 
            #                                 single_events_dict, 
            #                                 [channel_name], 
            #                                 visualization_config['events'],
            #                                 ax=axs[i, ax_number:ax_number+number_of_events])

            # print(f'Finished plotting for Channel: {channel_name} ({i+1}/{len(channel_labels)})')

        # Reduce figure size if it's too large
        fig_size_inches = fig.get_size_inches()
        max_size_inches = (100, 100)  # Adjust these values as needed
        if fig_size_inches[0] > max_size_inches[0] or fig_size_inches[1] > max_size_inches[1]:
            scale_factor = min(max_size_inches[0] / fig_size_inches[0], 
                               max_size_inches[1] / fig_size_inches[1])
            new_size = (fig_size_inches[0] * scale_factor, fig_size_inches[1] * scale_factor)
            fig.set_size_inches(new_size)

        plt.tight_layout()
        # fig.savefig(os.path.join(visualization_config['output_path'], f'combined_evaluation_{highest_correlation:.4f}_{model_name}_{weight_epoch}.pdf'), dpi=300, bbox_inches='tight')
        # plt.close()
        # Save as PNG with compression
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        # Open the image and further compress it
        img = Image.open(buf)
        img = img.convert('RGB')  # Convert to RGB to ensure compatibility        
        
        output_filename = os.path.join(visualization_config['output_path'], 
                                       f'combined_eval_{highest_correlation:.4f}_{model_name}_{weight_epoch}.webp')
        
        img.save(output_filename, 'WEBP', quality=85, method=6)
        
        plt.close()
        buf.close()

        if visualization_config['do_wandb']:
            # Log final metrics to wandb
            wandb.log({
                "final_test_correlation": highest_correlation,
                "weight_epoch": weight_epoch
            })

def main():
    set_seed(42)  # You can choose any integer as your seed

    # Make a new timestamp folder at OUTPUT_DIRECTORY for this run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_output_directory = os.path.join(OUTPUT_DIRECTORY, timestamp)
    os.makedirs(base_output_directory, exist_ok=True)

    # Define channels to use
    input_channel_names = list(NIRS_COORDS.keys())
    target_channel_names = list(EEG_COORDS.keys())

    
    # input_channel_names = list(EEG_COORDS.keys())
    # input_channel_names.remove('Cz')
    # target_channel_names = ['Cz']

    # Configuration
    data_config = {
        'tasks': ['task_name'],
        'input_channel_names': input_channel_names,
        'target_channel_names': target_channel_names,
        'target_sample_rate': 200,
        'input_t_min': -5,
        'input_t_max': 10,
        'target_t_min': 0,
        'target_t_max': 1,
        'tokenization': 'wavelets',  # or 'raw', 'fpca', 'wavelets'
        'target_filter_range': [0.5, 4], #, Delta: 0.5–4 Hz, Theta: 4–8 Hz, Alpha: 8-12 Hz, Beta: 12–30 Hz, Gamma: 30–100 Hz
        'redo_tokenization': False,
        'get_data_function': 'get_data_nirs_eeg', #'get_data_eeg_to_eeg', #get_data_nirs_eeg
        'input_parameters': 'cbci', # 'hbo', 'hbr', 'cbci' Only works for get_data_nirs_eeg
    }

    visualization_config = {
        'do_wandb': False,
        'do_plot': True,
        'do_save': True,
        'weight_epochs': [500],
        'channel_names': target_channel_names,  # Channels to visualize
        'events': ['2-back non-target', '2-back target', '3-back non-target', '3-back target'],
        'output_path': base_output_directory
    }

    model_configs = [
        {
        'name': 'rnn',
        'hidden_dim': 64,
        'spatial_encoding': True,
        'num_lstm_layers': 1,
        'bidirectional': False,
        'dropout': 0,
            'use_attention': False,
        },
    ]

    training_configs = [
        {
            'do_train': True,
            'do_load': False,
            'redo_train': True,
            'num_epochs': 100,
            'num_train_windows': 1000,
            'test_size': 0.15,
            'validation_size': 0.05,
            'learning_rate': 0.001,
        }
    ]

    token_sizes = [
        {
            'input_token_size': 20,
            'target_token_size': 30,
        },
    ]

    
    # from config.training_dictionaries import training_configs, token_sizes, rnn_model_configs, mlp_model_configs

    run_number = 1
    #subject_ids = list(range(1, 28))  # Subject IDs from 1 to 27
    subject_ids = [4]
    for subject_id in subject_ids:
        for token_size in token_sizes:
            redo_tokenization = True
            for model_config in model_configs:
                for training_config in training_configs:
                    model_name = f'{model_config["name"]}_{subject_id:02d}_n_chs_{len(data_config["input_channel_names"])}_{len(data_config["target_channel_names"])}_{run_number}'
                    # Timestamp for loop
                    output_directory = os.path.join(base_output_directory, model_name)
                    os.makedirs(output_directory, exist_ok=True)

                    print(f"Running model for subject {subject_id} with config:")
                    print(f"Model config: {model_config}")
                    print(f"Training config: {training_config}")
                    print(f"Token size: {token_size}")

                    data_config = data_config | token_size | {'redo_tokenization': redo_tokenization}
                    visualization_config['output_path'] = output_directory
                    save_config(subject_id, model_config, data_config, training_config, visualization_config)
                    
                    run_model(subject_id, model_name, model_config, data_config, training_config, visualization_config)
                    gc.collect()
                    torch.cuda.empty_cache()

                    run_number += 1

if __name__ == '__main__':
    main()
