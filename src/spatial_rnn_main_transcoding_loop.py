'''

'''

import sys
sys.path.insert(1, '../')

import os
import joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mne
from mne_nirs.experimental_design import make_first_level_design_matrix, create_boxcar
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix

from sklearn.metrics import r2_score

import torch
from torch.utils.data import DataLoader

import csv
from sklearn.metrics import r2_score

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

import gc

from config.Constants import *

from processing.Format_Data import grab_ordered_windows, grab_random_windows, find_indices, EEGfNIRSData
from tokenization.FPCA import perform_fpca_over_channels, get_fpca_dict, plot_explained_variance_over_dict
from tokenization.CCA import perform_cca_over_channels, get_cca_dict

from utilities.Read_Data import read_matlab_file, read_subjects_data
from models.Model_Utilities import predict_eeg, create_rnn, create_mlp, create_transformer
from models.Model_Utilities import PerChannelEncoder, Decoder, Seq2Seq, inference
from utilities.utilities import translate_channel_name_to_ch_id

from plotting.Windowed_Correlation import rolling_correlation

from processing.Processing_EEG import process_eeg_epochs
from processing.Processing_NIRS import process_nirs_epochs


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
loss_amounts = {
    'rnn': 0.01,
    'mlp': 0.001,
    'transformer': 0.01
}

def model_hrf_design_matrix(raw_haemo):
    design_matrix = make_first_level_design_matrix(
        raw_haemo,
        drift_model="cosine",
        high_pass=0.005,  # Must be specified per experiment
        hrf_model="spm",
        stim_dur=5.0,
    )

    print(f'Design Matrix: {design_matrix.shape}')
    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig = plot_design_matrix(design_matrix, ax=ax1)
    plt.show()

    return design_matrix

def convolve_hrf(raw_haemo, design_matrix):
    print(f'Plotting HRF')
    print(f'Haemo Size: {raw_haemo.get_data().shape}')
    events, single_events_dict = mne.events_from_annotations(raw_haemo)
    print(f'Event Times: {events[:-5]}')

    fig, ax = plt.subplots(constrained_layout=True)
    s = create_boxcar(raw_haemo, stim_dur=0.4)
    print(f's type: {type(s)}')
    print(f's: {np.shape(s)}')
    ax.plot(raw_haemo.times, s[:, 1])
    ax.plot(design_matrix["Tapping_Left"])
    ax.legend(["Stimulus", "Expected Response"])
    ax.set(xlim=(180, 300), xlabel="Time (s)", ylabel="Amplitude")

def run_glm(raw_haemo, design_matrix):
    print(f'Running GLM')
    data_subset = raw_haemo.copy().pick(picks=range(2))
    glm_est = run_glm(data_subset, design_matrix)
    glm_est.to_dataframe().head(9)

def get_hrf(subject_ids):
    for subject_id in subject_ids:
        _, raw_haemo = read_subjects_data(
                subjects=[f'VP{subject_id:03d}'],
                raw_data_directory=RAW_DIRECTORY,
                tasks=tasks,
                eeg_event_translations=EEG_EVENT_TRANSLATIONS,
                nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
                eeg_coords=EEG_COORDS,
                tasks_stimulous_to_crop=TASK_STIMULOUS_TO_CROP,
                trial_to_check_nirs=TRIAL_TO_CHECK_NIRS,
                eeg_t_min=eeg_t_min,
                eeg_t_max=eeg_t_max,
                nirs_t_min=nirs_t_min,
                nirs_t_max=nirs_t_max,
                eeg_sample_rate=eeg_sample_rate,
                redo_preprocessing=True,
            )
        
        print(f'Shape samples: {raw_haemo.get_data().shape}')
        events, single_events_dict = mne.events_from_annotations(raw_haemo)
        print(f'Event Times full: {events[-5:]}')
        aasdasds=asdasd
        path_to_design_matrix = os.path.join(OUTPUT_DIRECTORY, f'design_matrix_{subject_id}.csv')
        if not os.path.exists(path_to_design_matrix):
            design_matrix = model_hrf_design_matrix(raw_haemo)
            design_matrix.to_csv(path_to_design_matrix)
        else:
            print(f'Design Matrix exists, skipping {subject_id}')
            design_matrix = pd.read_csv(path_to_design_matrix)

        convolve_hrf(raw_haemo, design_matrix)
        run_glm(raw_haemo, design_matrix)
    

def plot_grand_average_matrix(events, channels, grand_average_dict, sampling_rate):
    # plot average erp comparison between mne and manual
    fig, axs = plt.subplots(len(events), len(channels), figsize=(10, 10))
    times = None
    for event_name in events:
        for channel in channels:
            i = events.index(event_name)
            j = channels.index(channel)
            # non-target
            non_erp =  grand_average_dict[f'{event_name} non-target'].copy().pick(channel)
            times = non_erp.times*sampling_rate
            non_erp = non_erp.data.T
            # target
            target_erp = grand_average_dict[f'{event_name} target'].copy().pick(channel).data.T
            # non-target - target
            difference_erp = non_erp - target_erp

            # plot
            axs[i,j].plot(times, non_erp, label='Non-Target')
            axs[i,j].plot(times, target_erp, label='Target')
            axs[i,j].plot(times, difference_erp, label='N-T')

            if i == 0:
                axs[i,j].set_title(f'{channel}')
            if j == 0:
                axs[i,j].set_ylabel(f'{event_name}')

            # if test_events.index(event_name) == 0 and test_channels.index(channel) == 0:
            axs[i,j].legend()

def plot_grand_average():
    # testing grand average
    grand_average_dict = {
                        'eeg': {'2-back non-target':[],
                                '2-back target':[],
                                '3-back non-target':[],
                                '3-back target':[]},
                        'nirs': {'2-back non-target':[],
                                '2-back target':[],
                                '3-back non-target':[],
                                '3-back target':[]}
                        }
    subject_ids = np.arange(1,27) # 1-27
    for subject in subject_ids:
        eeg_raw_mne, nirs_raw_mne = read_subjects_data(
            subjects=[f'VP{subject:03d}'],
            raw_data_directory=RAW_DIRECTORY,
            tasks=tasks,
            eeg_event_translations=EEG_EVENT_TRANSLATIONS,
            nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
            eeg_coords=EEG_COORDS,
            tasks_stimulous_to_crop=TASK_STIMULOUS_TO_CROP,
            trial_to_check_nirs=TRIAL_TO_CHECK_NIRS,
            eeg_t_min=eeg_t_min,
            eeg_t_max=eeg_t_max,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_sample_rate=eeg_sample_rate,
            redo_preprocessing=False,
        )

        for signal_name, task_dict in grand_average_dict.items():
            if signal_name == 'eeg':
                epochs = process_eeg_epochs(
                    eeg_raw_mne, 
                    eeg_t_min,
                    eeg_t_max)
            elif signal_name == 'nirs':
                epochs = process_nirs_epochs(
                    nirs_raw_mne, 
                    eeg_t_min,
                    eeg_t_max)

            for task_name in task_dict:
                grand_average_dict[signal_name][task_name].append(epochs[task_name].average())
        
    # get grand average
    grand_averages = {'eeg':{}, 'nirs':{}}
    for signal_name, task_dict in grand_average_dict.items():
        for task_name in task_dict:
            if len(task_dict[task_name]) != 0:
                grand_averages[signal_name][task_name] = mne.grand_average(grand_average_dict[signal_name][task_name])

    test_events = ['2-back',
                    '3-back']
    
    # EEG grand average
    eeg_test_channels = [
                'Cz',
                'Pz',]
    plot_grand_average_matrix(test_events, eeg_test_channels, grand_averages['eeg'], eeg_sample_rate)

    # NIRS grand average
    nirs_test_channels = [
                'AF7',
                'C3h',]
    
    nirs_channels_to_use_ids = translate_channel_name_to_ch_id(nirs_test_channels, NIRS_COORDS, nirs_raw_mne.ch_names)
    nirs_test_channels = []
    for channel_id in nirs_channels_to_use_ids:
        nirs_test_channels.append(f'{channel_id} hbo')
        nirs_test_channels.append(f'{channel_id} hbr')
    plot_grand_average_matrix(test_events, nirs_test_channels, grand_averages['nirs'], 10.42)

    plt.show()
    input('Press Enter to continue...')


def plot_erp_matrix(subject_ids,
                    nirs_test_channels=['AF7', 'C3h'],
                    eeg_test_channels=['Cz', 'Pz'],
                    test_events=['2-back non-target', '2-back target']):
    # Initialize ERP average dictionary
    erp_average_dict = {'eeg': {}, 'nirs': {}}
    for signal_name in erp_average_dict.keys():
        for task_name in test_events:
            erp_average_dict[signal_name][task_name] = []

    for subject in subject_ids:
        eeg_raw_mne, nirs_raw_mne = read_subjects_data(
            subjects=[f'VP{subject:03d}'],
            raw_data_directory=RAW_DIRECTORY,
            tasks=tasks,
            eeg_event_translations=EEG_EVENT_TRANSLATIONS,
            nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
            eeg_coords=EEG_COORDS,
            tasks_stimulous_to_crop=TASK_STIMULOUS_TO_CROP,
            trial_to_check_nirs=TRIAL_TO_CHECK_NIRS,
            eeg_t_min=eeg_t_min,
            eeg_t_max=eeg_t_max,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_sample_rate=eeg_sample_rate,
            redo_preprocessing=False,
        )
        for signal_name, task_dict in erp_average_dict.items():
            if signal_name == 'eeg':
                epochs = process_eeg_epochs(
                    eeg_raw_mne,
                    eeg_t_min,
                    eeg_t_max)
            elif signal_name == 'nirs':
                epochs = process_nirs_epochs(
                    nirs_raw_mne,
                    eeg_t_min,
                    eeg_t_max)
            for task_name in task_dict:
                erp_average_dict[signal_name][task_name].append(epochs[task_name].average())

    def create_plot(signal_type, test_channels):
        fig, axs = plt.subplots(len(test_channels), len(subject_ids), figsize=(20, 20))
        for i, channel in enumerate(test_channels):
            for j, subject in enumerate(subject_ids):
                for task_name in test_events:
                    erp_data = erp_average_dict[signal_type][task_name][j].copy().pick(channel).data.T
                    times = erp_average_dict[signal_type][task_name][j].times
                    axs[i, j].plot(times, erp_data, label=task_name)
                if i == 0:
                    axs[i, j].set_title(f'{subject}')
                if j == 0:
                    axs[i, j].set_ylabel(f'{channel}')
                if i == 0 and j == 0:
                    axs[i, j].legend()
        plt.suptitle(f'{signal_type.upper()} ERP Matrix')
        return fig

    root = tk.Tk()
    root.title("ERP Matrix")

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    nirs_channels_to_use_ids = translate_channel_name_to_ch_id(nirs_test_channels, NIRS_COORDS, nirs_raw_mne.ch_names)
    nirs_test_channels = []
    for channel_id in nirs_channels_to_use_ids:
        nirs_test_channels.append(f'{channel_id} hbo')
        nirs_test_channels.append(f'{channel_id} hbr')

    for signal_type, test_channels in [('eeg', eeg_test_channels), ('nirs', nirs_test_channels)]:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f'{signal_type.upper()} ERP Matrix')

        fig = create_plot(signal_type, test_channels)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        # Create a scrollable frame
        scrollable_frame = ttk.Frame(frame)
        scrollable_frame.pack(fill='both', expand=True)

        # Add canvas to scrollable frame
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add scrollbars
        x_scrollbar = ttk.Scrollbar(scrollable_frame, orient=tk.HORIZONTAL, command=canvas_widget.xview)
        y_scrollbar = ttk.Scrollbar(scrollable_frame, orient=tk.VERTICAL, command=canvas_widget.yview)
        canvas_widget.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add toolbar for zooming and panning
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()

def plot_erp(eeg_epochs, test_events, test_channels):
    # plot average erp comparison between mne and manual
    fig, axs = plt.subplots(len(test_events), len(test_channels), figsize=(10, 10))
    print(np.shape(axs))
    times = None
    for event_name in test_events:
        for channel in test_channels:
            epoch_eeg_select = eeg_epochs.copy().pick(channel)
            # non-target
            non_erp =  epoch_eeg_select[f'{event_name} non-target']
            times = non_erp.times
            non_erp = non_erp.average().data.T
            # target
            target_erp = epoch_eeg_select[f'{event_name} target'].average().data.T
            # non-target - target
            difference_erp = non_erp - target_erp
            # plot
            axs[test_events.index(event_name), test_channels.index(channel)].plot(times, non_erp, label='Non-Target')
            axs[test_events.index(event_name), test_channels.index(channel)].plot(times, target_erp, label='Target')
            axs[test_events.index(event_name), test_channels.index(channel)].plot(times, difference_erp, label='N-T')

            if test_events.index(event_name) == 0:
                axs[test_events.index(event_name), test_channels.index(channel)].set_title(f'{channel}')
            if test_channels.index(channel) == 0:
                axs[test_events.index(event_name), test_channels.index(channel)].set_ylabel(f'{event_name}')

            # if test_events.index(event_name) == 0 and test_channels.index(channel) == 0:
            axs[test_events.index(event_name), test_channels.index(channel)].legend()

def plot_erp_comparison(eeg_data, mrk_data, eeg_epochs, test_events, test_channels, single_events_dict):
    fig, axs = plt.subplots(len(test_events), len(test_channels), figsize=(20, 15))
    
    # Calculate ERP from extracted data
    for event_idx, event_name in enumerate(test_events):
        event_indices = mrk_data[mrk_data[:, 2] == single_events_dict[event_name], 0]
        
        for channel_idx, channel in enumerate(test_channels):
            channel_data = eeg_data[EEG_CHANNEL_NAMES.index(channel)]
            
            # Extract epochs
            epochs = []
            for event_time in event_indices:
                start = int(event_time + eeg_t_min * eeg_sample_rate)
                end = int(event_time + eeg_t_max * eeg_sample_rate)
                if start >= 0 and end < channel_data.shape[0]:
                    epochs.append(channel_data[start:end])
            
            # Calculate ERP
            erp = np.mean(epochs, axis=0)
            times = np.linspace(eeg_t_min, eeg_t_max, erp.shape[0])
            
            # Plot ERP from extracted data
            axs[event_idx, channel_idx].plot(times, erp, label='Extracted')
            axs[event_idx, channel_idx].set_title(f'{event_name} - {channel} (Extracted)')
            
            # Plot ERP from MNE epoched object
            evoked = eeg_epochs[event_name].average().pick(channel)
            axs[event_idx, channel_idx].plot(evoked.times, evoked.data.T, label='MNE')
            axs[event_idx, channel_idx].set_title(f'{event_name} - {channel} (MNE)')
            
            axs[event_idx, channel_idx].set_xlabel('Time (s)')
            axs[event_idx, channel_idx].set_ylabel('Amplitude')
            axs[event_idx, channel_idx].legend()
            axs[event_idx, channel_idx].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_erp_by_channel(train_eeg_data, 
                        train_mrk_data, 
                        single_events_dict, 
                        channel_names, 
                        event_selection, 
                        time_window = [-0.9, 1],
                        sampling_rate = 200):
    
    samples_window = [int(t * sampling_rate) for t in time_window]

    # reverse single_events_dict
    single_events_dict_reverse = {v: k for k, v in single_events_dict.items()}

    # Get unique markers for selected events
    unique_markers = [marker for marker, event in single_events_dict_reverse.items() if event in event_selection]
    print("Unique markers:", unique_markers)

    # Create figure and subplots
    n_cols = len(channel_names)
    fig, axs = plt.subplots(1, n_cols, figsize=(7*n_cols, 6), sharey=True)
    
    if n_cols == 1:
        axs = [axs]

    # Define a color cycle for different event types
    if len(unique_markers) > 2:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_markers)))
    else:
        # green. blue. purple
        colors = ['green', 'blue', 'purple']

    print(train_mrk_data)
    # Calculate and plot ERP for each channel
    for j, channel_name in enumerate(channel_names):
        channel_index = EEG_CHANNEL_NAMES.index(channel_name)

        avg_erp_list = []
        sem_erp_list = []
        for i, marker in enumerate(unique_markers):
            print(f"Processing {single_events_dict_reverse[marker]} for {channel_name}")

            # Find indices of the current marker
            marker_indices = np.where(train_mrk_data[:, 2] == marker)[0]
            marker_indices = train_mrk_data[marker_indices][:,0] # Grab sample index

            # Extract EEG data around each marker
            epochs = []
            for idx in marker_indices:
                start = idx + samples_window[0]
                end = idx + samples_window[1]
                if start >= 0 and end < train_eeg_data.shape[1]:
                    epochs.append(train_eeg_data[channel_index, start:end])
            
            epochs = np.array(epochs)
            print(f"Shape of epochs for {channel_name}, {single_events_dict_reverse[marker]}: {epochs.shape}")
            
            # Calculate average ERP and standard error of the mean
            avg_erp = np.mean(epochs, axis=0)
            sem_erp = np.std(epochs, axis=0) / np.sqrt(epochs.shape[0])  # SEM calculation
            avg_erp_list.append(avg_erp)
            sem_erp_list.append(sem_erp)
            print(f"Shape of avg_erp for {channel_name}, {single_events_dict_reverse[marker]}: {avg_erp.shape}")
            
            # Plot average ERP with standard error of the mean
            time = np.linspace(time_window[0], time_window[1], avg_erp.shape[0])
            axs[j].plot(time, avg_erp, color=colors[i], label=single_events_dict_reverse[marker])
            axs[j].fill_between(time, avg_erp - sem_erp, avg_erp + sem_erp, color=colors[i], alpha=0.2)
    
        # plot the difference of all markers
        diff_erp = None
        # diff_std = None
        for i in range(len(avg_erp_list)):
            if i == 0:
                diff_erp = avg_erp_list[i]
                # diff_std = std_erp_list[i]**2  # Variance
            else:
                diff_erp -= avg_erp_list[i]
                # diff_std += std_erp_list[i]**2  # Sum of variances
        
        # diff_std = np.sqrt(diff_std)  # Convert back to standard deviation

        axs[j].plot(time, diff_erp, color=colors[-1], label='Difference')
        # axs[j].fill_between(time, diff_erp- diff_std, diff_erp + diff_std, color=colors[-1], alpha=0.2) #

        axs[j].set_title(f'{channel_name} ERPs')
        axs[j].set_xlabel('Time (s)')
        axs[j].set_ylabel('Amplitude')
        axs[j].axvline(x=0, color='k', linestyle='--')  # Add vertical line at t=0
        axs[j].legend()

    plt.tight_layout()
    plt.show()

def compare_real_vs_predicted_erp(real_eeg_data, 
                                  predicted_eeg_data, 
                                  mrk_data, 
                                  single_events_dict, 
                                  channel_names, 
                                  event_selection,
                                  full_channel_names,
                                  time_window = [-0.1, 0.9],
                                  sampling_rate = 200,
                                  title = ''):
    
    samples_window = [int(t * sampling_rate) for t in time_window]

    # reverse single_events_dict
    single_events_dict_reverse = {v: k for k, v in single_events_dict.items()}

    # Get unique markers for selected events
    unique_markers = [marker for marker, event in single_events_dict_reverse.items() if event in event_selection]

    # Create figure and subplots
    n_cols = len(unique_markers)
    n_rows = len(channel_names)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows), sharey=True)
    
    if n_rows == 1:
        axs = [axs]
    if n_cols == 1:
        axs = [[ax] for ax in axs]

    # Define colors for real, predicted, and difference
    colors = ['green', 'blue', 'purple']

    print(mrk_data)
    # Calculate and plot ERP for each channel
    for j, channel_name in enumerate(channel_names):
        # channel_index = full_channel_names.index(channel_name)
        channel_index = j
        print(f'Channel Name: {channel_name}')

        for i, marker in enumerate(unique_markers):
            print(f"Processing {single_events_dict_reverse[marker]} for {channel_name}")

            # Find indices of the current marker
            marker_indices = np.where(mrk_data[:, 2] == marker)[0]
            marker_indices = mrk_data[marker_indices][:,0] # Grab sample index

            print(f'Marker Indices: {marker_indices}')
            print(f'Real EEG Data: {real_eeg_data.shape}')

            # Extract EEG data around each marker
            real_epochs = []
            predicted_epochs = []
            for idx in marker_indices:
                start = idx + samples_window[0]
                end = idx + samples_window[1]
                if start >= 0 and end < real_eeg_data.shape[1]:
                    real_epochs.append(real_eeg_data[channel_index, start:end])
                    predicted_epochs.append(predicted_eeg_data[channel_index, start:end])
            
            real_epochs = np.array(real_epochs)
            predicted_epochs = np.array(predicted_epochs)
            print(f"Shape of epochs for {channel_name}, {single_events_dict_reverse[marker]}: {real_epochs.shape}")
            
            # Calculate mean and SEM
            real_mean = np.mean(real_epochs, axis=0)
            real_sem = np.std(real_epochs, axis=0) / np.sqrt(real_epochs.shape[0])
            pred_mean = np.mean(predicted_epochs, axis=0)
            pred_sem = np.std(predicted_epochs, axis=0) / np.sqrt(predicted_epochs.shape[0])
            diff_mean = real_mean - pred_mean
            diff_sem = np.sqrt(real_sem**2 + pred_sem**2)  # Propagation of uncertainty
            
            # Plot mean and SEM
            time = np.linspace(time_window[0], time_window[1], real_mean.shape[0])
            axs[j][i].plot(time, real_mean, color=colors[0], label='Real')
            axs[j][i].fill_between(time, real_mean - real_sem, real_mean + real_sem, color=colors[0], alpha=0.3)
            axs[j][i].plot(time, pred_mean, color=colors[1], label='Predicted')
            axs[j][i].fill_between(time, pred_mean - pred_sem, pred_mean + pred_sem, color=colors[1], alpha=0.3)
            axs[j][i].plot(time, diff_mean, color=colors[2], label='Difference', alpha=0.5)
            # axs[j][i].fill_between(time, diff_mean - diff_sem, diff_mean + diff_sem, color=colors[2], alpha=0.1)
            
            axs[j][i].set_title(f'{channel_name} - {single_events_dict_reverse[marker]}')
            axs[j][i].set_xlabel('Time (s)')
            axs[j][i].set_ylabel('Amplitude')
            axs[j][i].axvline(x=0, color='k', linestyle='--')  # Add vertical line at t=0
            if j == 0 and i == 0:
                axs[j][i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f'{title}.jpeg'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def find_all_channel_pairs(eeg_coords, nirs_coords):
    eeg_positions = np.array(list(eeg_coords.values()))
    nirs_positions = np.array(list(nirs_coords.values()))
    
    distances = cdist(eeg_positions, nirs_positions)
    all_pairs = []
    
    for eeg_idx in range(distances.shape[0]):
        nirs_idx = np.argmin(distances[eeg_idx])
        eeg_channel = list(eeg_coords.keys())[eeg_idx]
        nirs_channel = list(nirs_coords.keys())[nirs_idx]
        distance = distances[eeg_idx, nirs_idx]
        all_pairs.append((eeg_channel, nirs_channel, distance))
    
    # Sort pairs by distance
    all_pairs.sort(key=lambda x: x[2])
    
    return all_pairs

def plot_channel_pairs(train_eeg_data, train_nirs_data, channel_pairs, eeg_channel_names, nirs_channel_names, chunk_size=10, eeg_fs=200, nirs_fs=10.42):
    n_pairs = len(channel_pairs)
    n_chunks = (n_pairs + chunk_size - 1) // chunk_size  # Round up division
    
    for chunk in range(n_chunks):
        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, n_pairs)
        current_pairs = channel_pairs[start:end]
        
        fig, axs = plt.subplots(len(current_pairs), 2, figsize=(15, 4*len(current_pairs)))
        
        for i, (eeg_channel, nirs_channel, distance) in enumerate(current_pairs):
            eeg_idx = eeg_channel_names.index(eeg_channel)
            nirs_idx = nirs_channel_names.index(nirs_channel)
            
            # Calculate time arrays
            eeg_time = np.arange(train_eeg_data.shape[1]) / eeg_fs
            nirs_time = np.arange(train_nirs_data.shape[1]) / nirs_fs
            
            # Plot EEG data
            axs[i, 0].plot(eeg_time, train_eeg_data[eeg_idx, :])
            axs[i, 0].set_title(f'EEG Channel: {eeg_channel}')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Amplitude')
            
            # Plot NIRS data
            axs[i, 1].plot(nirs_time, train_nirs_data[nirs_idx, :])
            axs[i, 1].set_title(f'NIRS Channel: {nirs_channel}')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('Amplitude')
            
            # Add distance information
            axs[i, 1].text(0.95, 0.95, f'Distance: {distance:.2f}', 
                           verticalalignment='top', horizontalalignment='right',
                           transform=axs[i, 1].transAxes, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'channel_pairs_chunk_{chunk+1}.png')
        plt.close()

def plot_eeg_nirs_comparison(train_eeg_data, train_nirs_data, eeg_sample_rate=200, nirs_sample_rate=10.42):
    # plot 5 closest closest eeg and nirs channels on the seperate subplots

    # Find all channel pairs sorted by distance
    all_pairs = find_all_channel_pairs(EEG_COORDS, NIRS_COORDS)

    # Plot the channel pairs in chunks of 10
    plot_channel_pairs(train_eeg_data, train_nirs_data, all_pairs, EEG_CHANNEL_NAMES, list(NIRS_COORDS.keys()), eeg_fs=eeg_sample_rate, nirs_fs=nirs_sample_rate)

    # Print all pairs for reference
    print("All EEG-NIRS channel pairs sorted by distance:")
    for eeg_channel, nirs_channel, distance in all_pairs:
        print(f"EEG: {eeg_channel} - NIRS: {nirs_channel} - Distance: {distance:.2f}")

def plot_scatter_between_timepoints(
    targets,
    predictions,
    channels_to_use,
):
    fig, axs = plt.subplots(len(channels_to_use), 1, figsize=(64, 100))
    for i in range(len(channels_to_use)):
        real_data = targets[:,i]
        predicted_data = predictions[:,i]

        # Create a color map from dark blue to light blue
        n_points = len(real_data)
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_points))

        # Plot the scatter with color gradient
        scatter = axs[i].scatter(real_data, predicted_data, c=range(n_points), cmap='Blues', alpha=0.6, s=0.5)

        axs[i].set_xlabel('Real')
        axs[i].set_ylabel('Predicted')
        axs[i].set_title(f'{channels_to_use[i]}')

        # Add a colorbar to show the gradient
        cbar = plt.colorbar(scatter, ax=axs[i])
        cbar.set_label('Point density')

    return fig, axs

from scipy.spatial.distance import cdist

def calculate_channel_distances(eeg_coords, nirs_coords):
    eeg_positions = np.array(list(eeg_coords.values()))
    nirs_positions = np.array(list(nirs_coords.values()))
    return cdist(nirs_positions, eeg_positions)

def run_model(subject_id_int,
              model_name_base, 
              nirs_channels_to_use_base, 
              eeg_channels_to_use, 
              eeg_channel_index, 
              nirs_channel_index, 
              num_epochs, 
              redo_train=False):
    subject_id = f'{subject_id_int:02d}'
    model_name = f'{model_name_base}_{subject_id}'
    final_model_path = os.path.join(MODEL_WEIGHTS, f'{model_name}_{num_epochs}.pth')
    if os.path.exists(final_model_path) and not redo_train:
        print(f'Model name exists, skipping {model_name}')
    else:
        print(f'Starting {model_name}')

        # Pre-allocate memory for training and testing data
        # eeg_data, nirs_data, mrk_data = read_matlab_file(subject_id, BASE_PATH)
        eeg_raw_mne, nirs_raw_mne = read_subjects_data(
            subjects=[f'VP{subject_id_int:03d}'],
            raw_data_directory=RAW_DIRECTORY,
            tasks=tasks,
            eeg_event_translations=EEG_EVENT_TRANSLATIONS,
            nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
            eeg_coords=EEG_COORDS,
            tasks_stimulous_to_crop=TASK_STIMULOUS_TO_CROP,
            trial_to_check_nirs=TRIAL_TO_CHECK_NIRS,
            eeg_t_min=eeg_t_min,
            eeg_t_max=eeg_t_max,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_sample_rate=eeg_sample_rate,
            redo_preprocessing=False,
        )

        fnirs_sample_rate = nirs_raw_mne.info['sfreq']

        #remove HEOG and VEOG
        eeg_raw_mne.drop_channels(['HEOG', 'VEOG'])
        # get only hbo
        nirs_raw_mne.pick(picks='hbo')

        mrk_data, single_events_dict = mne.events_from_annotations(eeg_raw_mne)
        mrk_data[:,0] -= mrk_data[0,0]
        mrk_data[:,0] += (1*eeg_sample_rate)
        print(single_events_dict)
        
        eeg_data = eeg_raw_mne.get_data()
        nirs_data = nirs_raw_mne.get_data()
        print(f'EEG Shape: {eeg_data.shape}') # n_channels x n_samples_eeg
        print(f'NIRS Shape: {nirs_data.shape}') # n_channels x n_samples_nirs
        print(f'MRK Shape: {mrk_data.shape}') # n_events x 3 (timestamp, event_type, event_id)

        # split train and test on eeg_data, nirs_data, and mrk_data
        test_size = int(eeg_data.shape[1]*test_size_in_subject)
        eeg_train_size = eeg_data.shape[1] - test_size
        test_size = int(nirs_data.shape[1]*test_size_in_subject)
        nirs_train_size = nirs_data.shape[1] - test_size

        train_eeg_data = eeg_data[:, :eeg_train_size]
        train_nirs_data = nirs_data[:, :nirs_train_size]

        test_eeg_data = eeg_data[:, eeg_train_size:]
        test_nirs_data = nirs_data[:, nirs_train_size:]

        # Calculate train and test mrk_data
        train_max_event_timestamp = train_eeg_data.shape[1]
        # MRK (timestamp, event_type, event_id)
        train_mrk_data = np.array([event for event in mrk_data if event[0] < train_max_event_timestamp])
        test_mrk_data = np.array([event for event in mrk_data if event[0] >= train_max_event_timestamp])
        # subtract train_max_event_timestamp from all index values in test_mrk_data
        test_mrk_data[:,0] -= train_max_event_timestamp

        # normalize train_eeg_data and test_eeg_data
        train_eeg_data = (train_eeg_data - np.mean(train_eeg_data)) / np.std(train_eeg_data)
        test_eeg_data = (test_eeg_data - np.mean(test_eeg_data)) / np.std(test_eeg_data)

        # normalize train_nirs_data and test_nirs_data
        train_nirs_data = (train_nirs_data - np.mean(train_nirs_data)) / np.std(train_nirs_data)
        test_nirs_data = (test_nirs_data - np.mean(test_nirs_data)) / np.std(test_nirs_data)

        print(train_mrk_data[:5])
        print(f'Train EEG Shape: {train_eeg_data.shape}')
        print(f'Train NIRS Shape: {train_nirs_data.shape}')
        print(f'Train MRK Shape: {train_mrk_data.shape}')
        print(f'Test EEG Shape: {test_eeg_data.shape}')
        print(f'Test NIRS Shape: {test_nirs_data.shape}')
        print(f'Test MRK Shape: {test_mrk_data.shape}')

        # print counts of unique markers in train_mrk_data and test_mrk_data
        reverse_single_events_dict = {v: k for k, v in single_events_dict.items()}
        print("Train MRK Counts:")
        train_unique, train_counts = np.unique(train_mrk_data[:, 2], return_counts=True)
        for marker, count in zip(train_unique, train_counts):
            print(f"  Marker {reverse_single_events_dict[marker]}: {count}")

        print("\nTest MRK Counts:")
        test_unique, test_counts = np.unique(test_mrk_data[:, 2], return_counts=True)
        for marker, count in zip(test_unique, test_counts):
            print(f"  Marker {reverse_single_events_dict[marker]}: {count}")

        # channel_names = ['Cz', 'Pz']  # Add more channel names as needed
        # event_selection = ['2-back non-target', '2-back target']
        # plot_erp_by_channel(train_eeg_data, train_mrk_data, single_events_dict, channel_names, event_selection)

        # Train data
        eeg_windowed_train, nirs_windowed_train, meta_data = grab_random_windows(
            nirs_data=train_nirs_data, 
            eeg_data=train_eeg_data,
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=eeg_sample_rate,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_t_min=eeg_t_min, 
            eeg_t_max=eeg_t_max,
            number_of_windows=1000)

        # Append to the preallocated arrays
        eeg_windowed_train = eeg_windowed_train.transpose(0,2,1)
        nirs_windowed_train = nirs_windowed_train.transpose(0,2,1)
    
        eeg_windowed_train = eeg_windowed_train[:,:, eeg_channel_index]
        nirs_windowed_train = nirs_windowed_train[:,:, nirs_channel_index]

        print(f'EEG Train Shape: {eeg_windowed_train.shape}')
        print(f'NIRS Train Shape: {nirs_windowed_train.shape}')

        # Train data in order for visualization
        eeg_windowed_train_ordered, nirs_windowed_train_ordered, meta_data, train_mrk_data_ordered = grab_ordered_windows(
            nirs_data=train_nirs_data, 
            eeg_data=train_eeg_data,
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=eeg_sample_rate,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_t_min=eeg_t_min, 
            eeg_t_max=eeg_t_max,
            markers=train_mrk_data)
        
        eeg_windowed_train_ordered = eeg_windowed_train_ordered.transpose(0,2,1)
        nirs_windowed_train_ordered = nirs_windowed_train_ordered.transpose(0,2,1)
    
        eeg_windowed_train_ordered = eeg_windowed_train_ordered[:,:, eeg_channel_index]
        nirs_windowed_train_ordered = nirs_windowed_train_ordered[:,:, nirs_channel_index]

        print(f'Train Ordered Shape: {eeg_windowed_train_ordered.shape}')
        print(f'Train Ordered Shape: {nirs_windowed_train_ordered.shape}')

        # Test data
        eeg_windowed_test, nirs_windowed_test, meta_data, test_mrk_data_ordered = grab_ordered_windows(
            nirs_data=test_nirs_data, 
            eeg_data=test_eeg_data,
            nirs_sampling_rate=fnirs_sample_rate,
            eeg_sampling_rate=eeg_sample_rate,
            nirs_t_min=nirs_t_min,
            nirs_t_max=nirs_t_max,
            eeg_t_min=eeg_t_min, 
            eeg_t_max=eeg_t_max,
            markers=test_mrk_data)

        print(f'EEG Test Shape: {eeg_windowed_test.shape}')
        print(f'NIRS Test Shape: {nirs_windowed_test.shape}')
        
        eeg_windowed_test = eeg_windowed_test.transpose(0,2,1)
        nirs_windowed_test = nirs_windowed_test.transpose(0,2,1)
    
        eeg_windowed_test = eeg_windowed_test[:,:, eeg_channel_index]
        nirs_windowed_test = nirs_windowed_test[:,:, nirs_channel_index]

        print(f'EEG Test Shape: {eeg_windowed_test.shape}')
        print(f'NIRS Test Shape: {nirs_windowed_test.shape}')

        # Prepare NIRS channel indices
        nirs_indices = torch.arange(len(nirs_channels_to_use_base))

        # Perform inference on test
        nirs_test_tensor = torch.from_numpy(nirs_windowed_test).float()
        eeg_test_tensor = torch.from_numpy(eeg_windowed_test).float()

        test_dataset = EEGfNIRSData(nirs_test_tensor, eeg_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Hyperparameters
        nirs_channels = len(nirs_channels_to_use_base)
        eeg_channels = len(eeg_channels_to_use)
        nirs_input_dim = 1  # Assuming one feature per NIRS channel (e.g., hemoglobin concentration)
        eeg_output_dim = 1  # Assuming one feature per EEG channel
        spatial_embedding_dim = 16
        encoder_hidden_size = 64
        decoder_hidden_size = 64
        num_layers = 1

        eeg_seq_len = eeg_windowed_train_ordered.shape[1]
        nirs_seq_len = nirs_windowed_train_ordered.shape[1]

        # Initialize spatial embeddings
        # nirs_spatial_embedding_net = SpatialEmbedding(input_dim=3, embedding_dim=spatial_embedding_dim).to(device)
        # eeg_spatial_embedding_net = SpatialEmbedding(input_dim=3, embedding_dim=spatial_embedding_dim).to(device)

        # Initialize encoder and decoder
        encoder = PerChannelEncoder(nirs_input_dim, encoder_hidden_size, num_layers).to(device)
        decoder = Decoder(eeg_output_dim, encoder_hidden_size, decoder_hidden_size, nirs_channels, nirs_seq_len, eeg_channels, num_layers).to(device)

        # Initialize Seq2Seq model
        model = Seq2Seq(encoder, decoder, device).to(device)

        def compute_distance_matrix(nirs_coords, eeg_coords):
            # nirs_coords shape: (nirs_channels, 3)
            # eeg_coords shape: (eeg_channels, 3)
    
            # Compute pairwise distances
            # distances = torch.cdist(nirs_coords.unsqueeze(0), eeg_coords.unsqueeze(0)).squeeze(0)

            nirs_coords = nirs_coords.unsqueeze(1)  # shape: (nirs_channels, 1, 3)
            eeg_coords = eeg_coords.unsqueeze(0)    # shape: (1, eeg_channels, 3)
            distances = torch.sqrt(torch.sum((nirs_coords - eeg_coords) ** 2, dim=2))  # shape: (nirs_channels, eeg_channels)
            return distances
        
        nirs_coords = np.array([value for key, value in NIRS_COORDS.items() if key in nirs_channels_to_use_base])
        nirs_coords = torch.from_numpy(nirs_coords).float()
        eeg_coords = np.array([value for key, value in EEG_COORDS.items() if key in eeg_channels_to_use])
        eeg_coords = torch.from_numpy(eeg_coords).float()

        distance_matrix = compute_distance_matrix(nirs_coords, eeg_coords)
        # Assuming distance_matrix shape: (nirs_channels, eeg_channels)
        distance_matrix_expanded = distance_matrix.unsqueeze(1).repeat(1, nirs_seq_len, 1)  # (nirs_channels, nirs_seq_len, eeg_channels)
        distance_matrix_expanded = distance_matrix_expanded.view(nirs_channels * nirs_seq_len, eeg_channels)  # (seq_len, eeg_channels)


        if do_train:
            nirs_train_tensor = nirs_windowed_train
            eeg_train_tensor = eeg_windowed_train
            
            nirs_train_tensor = torch.from_numpy(nirs_train_tensor).float()
            eeg_train_tensor = torch.from_numpy(eeg_train_tensor).float()
            meta_data_tensor = torch.from_numpy(np.array(meta_data)).float()
            
            print(nirs_train_tensor.shape)
            print(eeg_train_tensor.shape)
            
            dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
            latest_epoch = 0
            loss_list = []
            if do_load:
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
                print(f'Last loss: {float(loss_list[-1])/len(dataloader):.4f}')
        
            # Loss and optimizer
            criterion = torch.nn.MSELoss(reduction='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # nirs_spatial_embeddings = nirs_spatial_embedding_net(nirs_coords)  # shape: (nirs_channels, spatial_embedding_dim)
            # eeg_spatial_embeddings = eeg_spatial_embedding_net(eeg_coords)     # shape: (eeg_channels, spatial_embedding_dim)

            train_test_loss_dicct = {'train':[], 'test':[]}
            for epoch in range(latest_epoch, num_epochs):
                model.train()
                total_loss = 0
        
                for batch_idx, (nirs_inputs, eeg_targets) in enumerate(dataloader):
                    nirs_inputs = nirs_inputs.to(DEVICE).float()
                    eeg_targets = eeg_targets.to(DEVICE).float()
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(nirs_inputs, 
                                    eeg_targets, 
                                    distance_matrix_expanded, 
                                    teacher_forcing_ratio=0.5)

                    # Compute loss
                    # loss = criterion(outputs, eeg_targets)
                    # loss = criterion(outputs[:, 1:, :], eeg_targets[:, 1:, :])  # Ignore first time step

                    # Compute element-wise loss
                    loss = criterion(outputs, eeg_targets)  # Shape: (batch_size, eeg_seq_len, eeg_channels)

                    # Compute loss per channel by averaging over batch and sequence length
                    loss = loss.mean(dim=(0, 1))  # Shape: (eeg_channels,)

                    # Total loss is the mean over channels
                    loss = loss.mean()


                    loss.backward()
                    optimizer.step()
        
                    total_loss += loss.item()
                    # if (batch_idx+1) % 20 == 0 or batch_idx == 0:
                    #     print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}')
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():15f}')
                
                loss_list.append(total_loss)
        
                if (epoch+1) % 10 == 0:
                    # Save model weights
                    torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS, f'{model_name}_{epoch+1}.pth'))
                    with open(os.path.join(MODEL_WEIGHTS,f'loss_{model_name}_{epoch+1}.csv'), 'w', newline='') as file_ptr:
                        wr = csv.writer(file_ptr, quoting=csv.QUOTE_ALL)
                        wr.writerow(loss_list)
                    
                    # Assuming new_nirs_inputs is your new NIRS data
                    # nirs_spatial_embeddings and eeg_spatial_embeddings are computed as before
                    # distance_matrix is computed as before
                    # eeg_seq_len is defined based on your requirements

                    targets, predictions = inference(
                        model,
                        test_loader,
                        device,
                        distance_matrix_expanded
                    )

                    highest_r2 = 0
                    for channel_id in range(len(eeg_channels_to_use)):
                        targets_single = targets[:,:,channel_id]
                        predictions_single = predictions[:,:,channel_id]

                        target_average = np.mean(targets_single, axis=0)
                        prediction_average = np.mean(predictions_single, axis=0)
                        r2 = r2_score(target_average, prediction_average)
                        if r2 > highest_r2:
                            highest_r2 = r2
                
                    train_test_loss_dicct['train'].append(total_loss / len(dataloader))
                    train_test_loss_dicct['test'].append(highest_r2)
                    print(f'Epoch: {epoch+1}, Train Loss: {total_loss / len(dataloader):.10f}, Test Loss: {highest_r2:.10f}')
                elif (epoch+1) % 10 == 0:
                    print(f'Epoch: {epoch+1}, Average Loss: {total_loss / len(dataloader):.10f}')

            # Plotting train vs test loss
            fig, axs = plt.subplots(2, 1, figsize=(18, 6))
            axs[0].plot(train_test_loss_dicct['train'], label='Train Loss')
            axs[1].plot(train_test_loss_dicct['test'], label='Test R2')

            axs[0].set_xticklabels(np.arange(0, num_epochs+1, 50))
            axs[1].set_xticklabels(np.arange(0, num_epochs+1, 50))
            
            axs[1].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].set_title(f'Loss for {model_name}')
            axs[0].legend()
            axs[1].legend()
            axs[0].grid(True)
            axs[1].grid(True)
            fig.savefig(os.path.join(OUTPUT_DIRECTORY, f'loss_{model_name}.jpeg'))
            plt.close()

        # Perform inference on ordered train
        nirs_train_tensor = torch.from_numpy(nirs_windowed_train_ordered).float()
        eeg_train_tensor = torch.from_numpy(eeg_windowed_train_ordered).float()

        # Assuming fnirs_test and eeg_test are your test datasets
        train_dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        
        # Get weights for specific epoch
        weight_epochs = [10, 30, 50, 70]
        for weight_epoch in weight_epochs:
            model_path = f'{model_name}_{weight_epoch}.pth'
            model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
            model.to(DEVICE)

            model.eval()

            target_train, predictions_train = inference(
                        model,
                        train_loader,
                        device,
                        distance_matrix_expanded
                    )
            target_test, predictions_test = inference(
                        model,
                        test_loader,
                        device,
                        distance_matrix_expanded
                    )
            
            print(f'Weight Epoch: {weight_epoch}')
            print(f'Train Predictions Shape: {predictions_train.shape}')
            print(f'Train Targets Shape: {target_train.shape}')
            print(f'Test Predictions Shape: {predictions_test.shape}')
            print(f'Test Targets Shape: {target_test.shape}')
        
            # scipy.io.savemat(os.path.join(OUTPUT_DIRECTORY, f'test_{model_name}_{weight_epoch}.mat'), {'X': targets, 
            #                                                         'XPred':predictions,
            #                                                     'bins':10,
            #                                                     'scale':10,
            #                                       F              'srate':200})
            
            offset = 0  # Assuming no offset
            timeSigma = 100  # Assuming a given timeSigma
            num_bins = 50  # Assuming a given number of bins

            # Plotting target vs. output on concatenated data subplots for each channel
            highest_r2 = 0
            counter = 0
            fig, axs = plt.subplots(len(eeg_channels_to_use), 2, figsize=(128, 100))
            print(f'Plotting target vs output for {len(eeg_channels_to_use)} channels')
            for i in range(len(eeg_channels_to_use)):
                print(f'Channel Name: {eeg_channels_to_use[i]}')
                for j in range(2):
                    if j == 0:
                        type_label = 'Train'
                        do_legend = True
                        do_colorbar = False
                        targets_single = target_train[:,:,counter]
                        predictions_single = predictions_train[:,:,counter]

                        target_average = np.mean(targets_single, axis=0)
                        prediction_average = np.mean(predictions_single, axis=0)
                        r2 = r2_score(target_average, prediction_average)
                        
                        channel_label = list(EEG_COORDS.keys())[counter]
                    else:
                        type_label = 'Test'
                        do_legend = False
                        do_colorbar = True
                        targets_single = target_test[:,:,counter]
                        predictions_single = predictions_test[:,:,counter]

                        target_average = np.mean(targets_single, axis=0)
                        prediction_average = np.mean(predictions_single, axis=0)
                        r2 = r2_score(target_average, prediction_average)
                        if r2 > highest_r2:
                            highest_r2 = r2

                        channel_label = list(EEG_COORDS.keys())[counter]
                        counter += 1

                    # reshape to 1xn
                    targets_single = targets_single.reshape(1, -1)
                    predictions_single = predictions_single.reshape(1, -1)

                    chan_labels = [channel_label]  # Assuming label as provided in the command
                    test_fig = rolling_correlation(targets_single, 
                                                predictions_single, 
                                                chan_labels, 
                                                offset=offset,
                                                sampling_frequency=eeg_sample_rate,
                                                timeSigma=timeSigma, 
                                                num_bins=num_bins, 
                                                zoom_start=0, 
                                                #    zoom_end=500, 
                                                do_legend=do_legend,
                                                do_colorbar=do_colorbar,
                                                ax=axs[i, j], 
                                                title='')
                            
                    axs[i, j].text(0.5, 0.9, f'{eeg_channels_to_use[i]} {type_label} R-squared: {r2:.10f}', horizontalalignment='center', verticalalignment='center', transform=axs[i, j].transAxes)
            fig.savefig(os.path.join(OUTPUT_DIRECTORY, f'test_{highest_r2:.10f}_{model_name}_{weight_epoch}.jpeg'))
            plt.close()

            # concatenate windows targets and prredictions train (sample, window, channel) -> (sample*window, channel)
            targets_train = target_train.reshape(-1, len(eeg_channels_to_use))
            predictions_train = predictions_train.reshape(-1, len(eeg_channels_to_use))

            # plot scatter between timepoints in predicted vs real
            fig, axs = plot_scatter_between_timepoints(
                targets_train,
                predictions_train,
                eeg_channels_to_use
            )
            fig.savefig(os.path.join(OUTPUT_DIRECTORY, f'train_{highest_r2:.10f}_{model_name}_{weight_epoch}_scatter.jpeg'))
            plt.close()

            channel_names = eeg_channels_to_use # Add more channel names as needed
            event_selection = ['2-back non-target', '2-back target', '3-back non-target', '3-back target']
            # Train predicted vs real ERP from train_mrk
            print('Train ERP')
            compare_real_vs_predicted_erp(targets_train.T, 
                                          predictions_train.T, 
                                          train_mrk_data_ordered, 
                                          single_events_dict, 
                                          channel_names, 
                                          event_selection, 
                                          full_channel_names=EEG_CHANNEL_NAMES,
                                          title=f'train_{highest_r2:.10f}_{model_name}_{weight_epoch}_erp.jpeg')

            # concatenate windows targets and prredictions test (sample, window, channel) -> (sample*window, channel)
            targets_test = target_test.reshape(-1, len(eeg_channels_to_use))
            predictions_test = predictions_test.reshape(-1, len(eeg_channels_to_use))

            # plot scatter between timepoints in predicted vs real
            fig, axs = plot_scatter_between_timepoints(
                targets_test,
                predictions_test,
                eeg_channels_to_use
            )
            fig.savefig(os.path.join(OUTPUT_DIRECTORY, f'test_{highest_r2:.10f}_{model_name}_{weight_epoch}_scatter.jpeg'))
            plt.close()

            # Test predicted vs real ERP from test_mrk
            print('Test ERP')
            compare_real_vs_predicted_erp(targets_test.T, 
                                          predictions_test.T, 
                                          test_mrk_data_ordered, 
                                          single_events_dict, 
                                          channel_names, 
                                          event_selection, 
                                          full_channel_names=EEG_CHANNEL_NAMES,
                                          title=f'train_{highest_r2:.10f}_{model_name}_{weight_epoch}_erp.jpeg')



def main():
    # get_hrf(subject_ids)
    # input('Press Enter to continue...')

    # plot_grand_average()
    # plot_erp_matrix(subject_ids=subject_ids,
    #                 nirs_test_channels=list(NIRS_COORDS.keys()),
    #                 eeg_test_channels=EEG_CHANNEL_NAMES,
    #                 test_events=['2-back non-target', '2-back target'])

    # Define channels to use
    nirs_channels_to_use_base = list(NIRS_COORDS.keys())[:10]
    nirs_channel_index = find_indices(list(NIRS_COORDS.keys()),nirs_channels_to_use_base)

    eeg_channels_to_use = EEG_CHANNEL_NAMES[:3] + ['Cz', 'Pz']
    eeg_channel_index = find_indices(EEG_CHANNEL_NAMES,eeg_channels_to_use)

    for subject_id_int in subject_ids:
        for model_name_base in ['seq2seq']:
            run_model(subject_id_int, 
                    model_name_base, 
                    nirs_channels_to_use_base, 
                    eeg_channels_to_use, 
                    eeg_channel_index, 
                    nirs_channel_index, 
                    num_epochs, 
                    redo_train=False)
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    ## Subject/Trial Parameters ##
    subject_ids = np.arange(1,27) # 1-27
    subjects = []
    for i in subject_ids:
        subjects.append(f'VP{i:03d}')

    tasks = ['nback']
    hemoglobin_types = ['hbo', 'hbr']

    # NIRS Sampling rate
    # fnirs_sample_rate = 10.41
    # EEG Downsampling rate
    eeg_sample_rate = 200
    # Time window (seconds)
    eeg_t_min = 0
    eeg_t_max = 1
    nirs_t_min = -5
    nirs_t_max = 10
    offset_t = 0

    # Redo preprocessing pickle files, TAKES A LONG TIME 
    redo_preprocessing = False
    do_load = False
    do_train = False

    # data projection
    # nirs_token_size = 10
    # eeg_token_size = 5

    # training loop
    num_epochs = 100
    test_size_in_subject = 0.2 # percent of test data

    main()

