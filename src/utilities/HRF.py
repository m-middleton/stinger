import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

from mne_nirs.experimental_design import make_first_level_design_matrix, create_boxcar
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix

from utilities.utilities import translate_channel_name_to_ch_id

from config.Constants import *

from processing.Processing_EEG import process_eeg_epochs
from processing.Processing_NIRS import process_nirs_epochs

from utilities.Read_Data import read_matlab_file, read_subjects_data


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

def get_hrf(subject_ids, 
            tasks, 
            eeg_t_min, 
            eeg_t_max, 
            nirs_t_min, 
            nirs_t_max, 
            eeg_sample_rate):
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

def plot_grand_average(
        tasks,
        eeg_t_min,
        eeg_t_max,
        nirs_t_min,
        nirs_t_max,
        eeg_sample_rate
):
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
