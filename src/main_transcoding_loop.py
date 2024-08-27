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
from processing.FPCA import perform_fpca_over_channels, get_fpca_dict, plot_explained_variance_over_dict
from processing.CCA import perform_cca_over_channels, get_cca_dict

from utilities.Read_Data import read_matlab_file, read_subjects_data
from utilities.Model_Utilities import predict_eeg, create_rnn, create_mlp, create_transformer
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
    fig, ax = plt.subplots(constrained_layout=True)
    s = create_boxcar(raw_haemo, stim_dur=5.0)
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
                redo_preprocessing=False,
            )
        
        design_matrix = model_hrf_design_matrix(raw_haemo)
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

def run_model(subject_id_int,
              model_name_base, 
              nirs_channels_to_use_base, 
              eeg_channels_to_use, 
              eeg_channel_index, 
              nirs_channel_index, 
              num_epochs, 
              redo_train=False):
    model_function = model_functions[model_name_base]
    loss_amount = loss_amounts[model_name_base]
    subject_id = f'{subject_id_int:02d}'
    model_name = f'{model_name_base}_{subject_id}'
    final_model_path = os.path.join(MODEL_WEIGHTS, f'{model_name}_{num_epochs}.pth')
    if os.path.exists(final_model_path) and not redo_train:
        print(f'Model name exists, skipping {model_name}')
    else:
        print(f'Starting {model_name}')
        model = model_function(len(nirs_channels_to_use_base)*nirs_token_size, len(eeg_channels_to_use)*eeg_token_size)
            
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

        mrk_data, single_events_dict = mne.events_from_annotations(eeg_raw_mne)
        reverse_events_dict = {v: k for k, v in single_events_dict.items()}
        test_events = ['2-back',
                        '3-back']
        test_channels = ['Pz',
                    'Cz',]
        
        print(single_events_dict)

        eeg_epochs = process_eeg_epochs(
                eeg_raw_mne, 
                eeg_t_min,
                eeg_t_max)
        
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

        # eeg_data = eeg_raw_mne.get_data()
        # nirs_data = nirs_raw_mne.get_data()
        # print(f'EEG Shape: {eeg_data.shape}') # n_channels x n_samples_eeg
        # print(f'NIRS Shape: {nirs_data.shape}') # n_channels x n_samples_nirs
        # print(f'MRK Shape: {mrk_data.shape}') # n_events x 3 (timestamp, event_type, event_id)
            
        # Get channel index for Cz from EEG_COORDS dict
        # test_event_ids = [single_events_dict[event] for event in test_events]
        # channel_index = EEG_CHANNEL_NAMES.index(test_channel)
        # channel_index = 0

        # test_eeg_data = eeg_raw_mne.copy().pick_channels([test_channel]).get_data()
        # print(f'single channel shape {test_eeg_data.shape}')
        # # plot eeg erp around each event id manually
        # for event_id in test_event_ids:
        #     event_id = int(event_id)
        #     event_data = mrk_data[mrk_data[:,2] == event_id]
        #     print(f'Event shape {event_data.shape}')

        #     event_list = []
        #     for event in event_data:
        #         eeg_samples_min = event[0] + (eeg_t_min*eeg_sample_rate)
        #         eeg_samples_max = event[0] + (eeg_t_max*eeg_sample_rate)
        #         # Get baseline
        #         baseline = test_eeg_data[:, eeg_samples_min:event[0]]
        #         event_data = test_eeg_data[:, eeg_samples_min:eeg_samples_max+1]
        #         corrected_data = event_data - np.mean(baseline, axis=1).reshape(-1, 1)
        #         event_list.append(corrected_data)
        #     single_erp_data = np.array(event_list)

        #     mean_data = np.mean(single_erp_data[:,channel_index,:], axis=0)
        #     print(f'ERP manual {single_erp_data.shape}')
        #     print(f'ERP manual {mean_data.shape}')

        #     axs[1].plot(times, mean_data)
        #     axs[1].set_title(f'EEG ERP for Event ID: {event_id}')
        #     print(f'Event ID: {reverse_events_dict[event_id]}')

        plt.show()

        input('Press Enter to continue...')
        asdasd=asdasd

        # split train and test on eeg_data, nirs_data, and mrk_data
        test_size = int(eeg_data.shape[1]*test_size_in_subject)
        train_size = eeg_data.shape[1] - test_size

        train_eeg_data = eeg_data[:, :train_size]
        train_nirs_data = nirs_data[:, :train_size]
        # train_mrk_data = mrk_data[:, :train_size]

        test_eeg_data = eeg_data[:, train_size:]
        test_nirs_data = nirs_data[:, train_size:]
        # test_mrk_data = mrk_data[:, train_size:]

        # get CA dict
        # cca_dict = get_cca_dict(subject_id, train_nirs_data, train_eeg_data, token_size)
        fpca_dict = get_fpca_dict(subject_id, 
                                train_nirs_data, 
                                train_eeg_data, 
                                nirs_token_size,
                                eeg_token_size,
                                model_weights=MODEL_WEIGHTS,
                                nirs_t_min=nirs_t_min,
                                nirs_t_max=nirs_t_max,
                                eeg_t_min=eeg_t_min,
                                eeg_t_max=eeg_t_max)
        eeg_fpcas = fpca_dict['eeg']
        nirs_fpcas = fpca_dict['nirs']

        # plot variance explained
        plot_explained_variance_over_dict(nirs_fpcas, path=os.path.join(OUTPUT_DIRECTORY, f'variance_{model_name}_nirs_fpca.jpeg'), channel_names=nirs_channels_to_use_base)
        plot_explained_variance_over_dict(eeg_fpcas, path=os.path.join(OUTPUT_DIRECTORY, f'variance_{model_name}_eeg_fpca.jpeg'), channel_names=eeg_channels_to_use)

        # Train data
        eeg_windowed_train, nirs_windowed_train, meta_data = grab_random_windows(
                        nirs_data=train_nirs_data, 
                        eeg_data=train_eeg_data,
                        sampling_rate=200,
                        nirs_t_min=nirs_t_min, 
                        nirs_t_max=nirs_t_max,
                        eeg_t_min=0, 
                        eeg_t_max=1,
                        number_of_windows=10000)
        
        eeg_windowed_train = eeg_windowed_train[:, :, :eeg_lookback]
        nirs_windowed_train = nirs_windowed_train[:, :, :fnirs_lookback]

        # nirs_windowed_train = perform_cca_over_channels(nirs_windowed_train, cca_dict, token_size)
        # eeg_windowed_train = perform_cca_over_channels(eeg_windowed_train, cca_dict, token_size)
        nirs_windowed_train = perform_fpca_over_channels(nirs_windowed_train, 
                                                        nirs_fpcas, 
                                                        nirs_token_size)
        eeg_windowed_train = perform_fpca_over_channels(eeg_windowed_train, 
                                                        eeg_fpcas, 
                                                        eeg_token_size)

        n_channels = nirs_windowed_train.shape[1]
        # # plot channels
        # fig, axs = plt.subplots(n_channels, 1, figsize=(10, 10))
        # for i in range(n_channels):
        #     axs[i].plot(nirs_windowed_train[0, i, :])
        # plt.show()

        # Append to the preallocated arrays
        eeg_windowed_train = eeg_windowed_train.transpose(0,2,1)
        nirs_windowed_train = nirs_windowed_train.transpose(0,2,1)
    
        eeg_windowed_train = eeg_windowed_train[:,:, eeg_channel_index]
        nirs_windowed_train = nirs_windowed_train[:,:, nirs_channel_index]

        print(f'EEG Train Shape: {eeg_windowed_train.shape}')
        print(f'NIRS Train Shape: {nirs_windowed_train.shape}')

        # Train data in order for visualization
        eeg_windowed_train_ordered, nirs_windowed_train_ordered, meta_data = grab_ordered_windows(
            nirs_data=train_nirs_data, 
            eeg_data=train_eeg_data,
            sampling_rate=200,
            nirs_t_min=nirs_t_min, 
            nirs_t_max=nirs_t_max,
            eeg_t_min=0, 
            eeg_t_max=1)
    
        eeg_windowed_train = eeg_windowed_train[:, :, :eeg_lookback]
        nirs_windowed_train = nirs_windowed_train[:, :, :fnirs_lookback]
                
        # nirs_windowed_test = perform_cca_over_channels(nirs_windowed_test, cca_dict, token_size)
        nirs_windowed_train_ordered = perform_fpca_over_channels(nirs_windowed_train_ordered, nirs_fpcas, nirs_token_size)
        
        eeg_windowed_train_ordered = eeg_windowed_train_ordered.transpose(0,2,1)
        nirs_windowed_train_ordered = nirs_windowed_train_ordered.transpose(0,2,1)
    
        eeg_windowed_train_ordered = eeg_windowed_train_ordered[:,:, eeg_channel_index]
        nirs_windowed_train_ordered = nirs_windowed_train_ordered[:,:, nirs_channel_index]

        print(f'EEG Train Ordered Shape: {eeg_windowed_train_ordered.shape}')
        print(f'NIRS Train Ordered Shape: {nirs_windowed_train_ordered.shape}')

        # Test data
        eeg_windowed_test, nirs_windowed_test, meta_data = grab_ordered_windows(
                    nirs_data=test_nirs_data, 
                    eeg_data=test_eeg_data,
                    sampling_rate=200,
                    nirs_t_min=nirs_t_min,
                    nirs_t_max=nirs_t_max,
                    eeg_t_min=0, 
                    eeg_t_max=1)
        
        eeg_windowed_train = eeg_windowed_train[:, :, :eeg_lookback]
        nirs_windowed_train = nirs_windowed_train[:, :, :fnirs_lookback]

        # nirs_windowed_test = perform_cca_over_channels(nirs_windowed_test, cca_dict, token_size)
        nirs_windowed_test = perform_fpca_over_channels(nirs_windowed_test, 
                                                        nirs_fpcas, 
                                                        nirs_token_size)
        
        eeg_windowed_test = eeg_windowed_test.transpose(0,2,1)
        nirs_windowed_test = nirs_windowed_test.transpose(0,2,1)
    
        eeg_windowed_test = eeg_windowed_test[:,:, eeg_channel_index]
        nirs_windowed_test = nirs_windowed_test[:,:, nirs_channel_index]

        print(f'EEG Test Shape: {eeg_windowed_test.shape}')
        print(f'NIRS Test Shape: {nirs_windowed_test.shape}')

        # Perform inference on test
        nirs_test_tensor = nirs_windowed_test.reshape(-1, len(nirs_channel_index)*nirs_token_size)

        nirs_test_tensor = torch.from_numpy(nirs_test_tensor).float()
        eeg_test_tensor = torch.from_numpy(eeg_windowed_test).float()

        test_dataset = EEGfNIRSData(nirs_test_tensor, eeg_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if do_train:
            # flatten channels and tokens
            nirs_train_tensor = nirs_windowed_train.reshape(-1, len(nirs_channel_index)*nirs_token_size)
            eeg_train_tensor = eeg_windowed_train.reshape(-1, len(eeg_channel_index)*eeg_token_size)
            
            nirs_train_tensor = torch.from_numpy(nirs_train_tensor).float()
            eeg_train_tensor = torch.from_numpy(eeg_train_tensor).float()
            meta_data_tensor = torch.from_numpy(np.array(meta_data)).float()
            
            print(nirs_train_tensor.shape)
            print(eeg_train_tensor.shape)
            
            dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
            dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
        
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
        
            model.to(DEVICE)
        
            # Optimizer and loss function
            optimizer = Adam(model.parameters(), lr=loss_amount)
            loss_function = torch.nn.MSELoss()
            train_test_loss_dicct = {'train':[], 'test':[]}
            for epoch in range(latest_epoch, num_epochs):
                model.train()
                total_loss = 0
        
                for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                    X_batch = X_batch.to(DEVICE).float()
                    y_batch = y_batch.to(DEVICE).float()
                    
                    # Forward pass
                    predictions = model(X_batch)
        
                    # Loss calculation
                    loss = loss_function(predictions, y_batch)
        
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                    total_loss += loss.item()
                    # if (batch_idx+1) % 20 == 0 or batch_idx == 0:
                    #     print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}')
                
                loss_list.append(total_loss)
        
                if (epoch+1) % 50 == 0:
                    # Save model weights
                    torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS, f'{model_name}_{epoch+1}.pth'))
                    with open(os.path.join(MODEL_WEIGHTS,f'loss_{model_name}_{epoch+1}.csv'), 'w', newline='') as file_ptr:
                        wr = csv.writer(file_ptr, quoting=csv.QUOTE_ALL)
                        wr.writerow(loss_list)
                    
                    targets, predictions = predict_eeg(model, 
                                                    test_loader, 
                                                    n_samples=eeg_windowed_test.shape[0], 
                                                    n_channels=eeg_windowed_test.shape[2], 
                                                    n_lookback=eeg_windowed_test.shape[1],
                                                    eeg_token_size=eeg_token_size,
                                                    eeg_fpcas=eeg_fpcas)
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
                    print(f'Epoch: {epoch+1}, Train Loss: {total_loss / len(dataloader):.4f}, Test Loss: {highest_r2:.4f}')
                elif (epoch+1) % 10 == 0:
                    print(f'Epoch: {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}')

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
        nirs_train_tensor = nirs_windowed_train_ordered.reshape(-1, len(nirs_channel_index)*nirs_token_size)
        nirs_train_tensor = torch.from_numpy(nirs_train_tensor).float()
        eeg_train_tensor = torch.from_numpy(eeg_windowed_train_ordered).float()

        # Assuming fnirs_test and eeg_test are your test datasets
        train_dataset = EEGfNIRSData(nirs_train_tensor, eeg_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        
        # Get weights for specific epoch
        weight_epochs = [100, 250, 500, 1000]
        for weight_epoch in weight_epochs:
            model_path = f'{model_name}_{weight_epoch}.pth'
            model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS, model_path)))
            model.to(DEVICE)
            model.eval()

            target_train, predictions_train = predict_eeg(model, 
                                            train_loader, 
                                            n_samples=eeg_windowed_train_ordered.shape[0], 
                                            n_channels=eeg_windowed_train_ordered.shape[2], 
                                            n_lookback=eeg_windowed_train_ordered.shape[1],
                                            eeg_token_size=eeg_token_size,
                                            eeg_fpcas=eeg_fpcas)
            targets, predictions = predict_eeg(model, 
                                            test_loader, 
                                            n_samples=eeg_windowed_test.shape[0], 
                                            n_channels=eeg_windowed_test.shape[2], 
                                            n_lookback=eeg_windowed_test.shape[1],
                                            eeg_token_size=eeg_token_size,
                                            eeg_fpcas=eeg_fpcas)
            
            print(f'Weight Epoch: {weight_epoch}')
            print(f'Predictions Shape: {predictions.shape}')
            print(f'Targets Shape: {targets.shape}')
        
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
            fig, axs = plt.subplots(len(eeg_channels_to_use), 2, figsize=(18, 100))
            for i in range(len(eeg_channels_to_use)):
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
                        targets_single = targets[:,:,counter]
                        predictions_single = predictions[:,:,counter]

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
                                                sampling_frequency=eeg_lookback,
                                                timeSigma=timeSigma, 
                                                num_bins=num_bins, 
                                                zoom_start=0, 
                                                #    zoom_end=500, 
                                                do_legend=do_legend,
                                                do_colorbar=do_colorbar,
                                                ax=axs[i, j], 
                                                title='')
                            
                    axs[i, j].text(0.5, 0.9, f'{eeg_channels_to_use[i]} {type_label} R-squared: {r2:.4f}', horizontalalignment='center', verticalalignment='center', transform=axs[i, j].transAxes)
            fig.savefig(os.path.join(OUTPUT_DIRECTORY, f'test_{highest_r2:.4f}_{model_name}_{weight_epoch}.jpeg'))
            plt.close()

def main():
    
    # subject_ids = np.arange(1, 27)  # 1-27
    subject_ids = [1]

    get_hrf(subject_ids)
    input('Press Enter to continue...')

    # plot_grand_average()
    # plot_erp_matrix(subject_ids=subject_ids,
    #                 nirs_test_channels=list(NIRS_COORDS.keys()),
    #                 eeg_test_channels=EEG_CHANNEL_NAMES,
    #                 test_events=['2-back non-target', '2-back target'])

    # Define channels to use
    # nirs_channels_to_use_base = list(NIRS_COORDS.keys())
    # nirs_channel_index = find_indices(list(NIRS_COORDS.keys()),nirs_channels_to_use_base)

    # eeg_channels_to_use = EEG_CHANNEL_NAMES
    # eeg_channel_index = find_indices(EEG_CHANNEL_NAMES,eeg_channels_to_use)

    # for subject_id_int in subject_ids[8:]:
    #     for model_name_base in ['rnn', 'mlp']:
    #         run_model(subject_id_int, 
    #                 model_name_base, 
    #                 nirs_channels_to_use_base, 
    #                 eeg_channels_to_use, 
    #                 eeg_channel_index, 
    #                 nirs_channel_index, 
    #                 num_epochs, 
    #                 redo_train=False)
    #         gc.collect()
    #         torch.cuda.empty_cache()

if __name__ == '__main__':
    ## Subject/Trial Parameters ##
    subject_ids = np.arange(1,27) # 1-27
    subjects = []
    for i in subject_ids:
        subjects.append(f'VP{i:03d}')

    tasks = ['nback']
    hemoglobin_types = ['hbo', 'hbr']

    # NIRS Sampling rate
    fnirs_sample_rate = 10.41
    # EEG Downsampling rate
    eeg_sample_rate = 200
    # Time window (seconds)
    eeg_t_min = -0.5
    eeg_t_max = 1
    nirs_t_min = -0.5
    nirs_t_max = 1
    offset_t = 0

    # Redo preprocessing pickle files, TAKES A LONG TIME 
    redo_preprocessing = False
    do_load = False
    do_train = True

    # data projection
    nirs_token_size = 10
    eeg_token_size = 5
    fnirs_lookback = 4000
    eeg_lookback = 200

    # training loop
    num_epochs = 1000
    test_size_in_subject = 0.2 # percent of test data

    main()

