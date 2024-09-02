'''
This file contains functions to read data from the .mat files and the .vhdr files
'''

import numpy as np
from scipy.stats import linregress
import os

import mne
from scipy.io import loadmat
import pickle

from processing.Processing_EEG import process_eeg_raw
from processing.Processing_NIRS import process_nirs_raw

def read_matlab_file(subject_id, base_path):
    '''Read matlab file and return data'''
    subject_data = loadmat(os.path.join(base_path, 'matfiles', f'data_vp0{subject_id}.mat'))['subject_data_struct'][0]
    # eeg subject_data[1][0]
    eeg_data = []
    for session_eeg_data in subject_data[1][0]:
        eeg_data.append(session_eeg_data.T)
    eeg_data = np.hstack(eeg_data)
    # fnirs subject_data[3][0]
    nirs_data = []
    for session_nirs_data in subject_data[3][0]:
        nirs_data.append(session_nirs_data.T)
    nirs_data = np.hstack(nirs_data)
    # mrk subject_data[5][0]
    mrk_data = []
    for session_mrk_data in subject_data[5][0]:
        mrk_data.append(session_mrk_data.T)
    mrk_data = np.hstack(mrk_data)

    assert eeg_data.shape[1] == nirs_data.shape[1]

    return eeg_data, nirs_data, mrk_data

def read_raw_eeg(file_name):
    return mne.io.read_raw_brainvision(file_name, 
                                eog=('HEOG', 'VEOG'), # Designate EOG channels
                                misc='auto', 
                                scale=1.0, 
                                preload=True, 
                                verbose=None)

def read_raw_nirs(file_name):
    return mne.io.read_raw_nirx(file_name, 
                                #saturated='ignore', 
                                preload=True, 
                                verbose=None)

def read_subject_raw_nirs(
        root_directory,
        tasks_to_do,
        trial_to_check={}, 
        nirs_event_translations={},
        eeg_translation_events_dict={},
        task_stimulous_to_crop={}):
    '''
    Read the raw nirs data for a subject
    input:
        root_directory (str)
        tasks_to_do (list of str)
        trial_to_check (dictionary of lists of str)
        nirs_event_translations (dictionary of dictionaries)
        eeg_translation_events_dict (dictionary of dictionaries)
        task_stimulous_to_crop (dictionary of lists of str)
    '''
    raw_intensities = []
    counter = 0

    raw_dict = {task: {} for task in trial_to_check.keys()}
    for subdir, dirs, files in os.walk(root_directory):
        for task_name, session_names in trial_to_check.items():
            if any(f in subdir for f in session_names) and task_name in tasks_to_do:
                print(subdir)
                raw_intensity = read_raw_nirs(subdir)

                folder_name = os.path.basename(os.path.normpath(subdir))
                if len(eeg_translation_events_dict) > 0:
                    session_index = session_names.index(folder_name)
                    eeg_events = eeg_translation_events_dict[task_name][session_index]['events'].copy()
                    event_translations = eeg_translation_events_dict[task_name][session_index]['translations'].copy()

                    # Transform events to fNIRS time
                    original_events, single_events_dict = mne.events_from_annotations(raw_intensity)
                    for v,k in single_events_dict.items():
                        if v in nirs_event_translations[task_name]:
                            original_events[:,2][original_events[:,2] == k] = event_translations[nirs_event_translations[task_name][v]]
                        else:
                            original_events = np.delete(original_events, np.where(original_events[:,2] == k), axis=0)

                    check_not_same_events = np.isin(eeg_events[:,2], original_events[:,2])
                    eeg_events_tmp = eeg_events[check_not_same_events]

                    fnirs_time = np.array(original_events[:,0])/raw_intensity.info['sfreq']
                    eeg_time = np.array(eeg_events_tmp[:,0])/1000

                    # Get time offset coeficients
                    lr = linregress(eeg_time, fnirs_time)
                    m_1=lr.slope
                    c_1=lr.intercept
    
                    events, single_events_dict = mne.events_from_annotations(raw_intensity)

                    # Add translated eeg events
                    eeg_events_tmp = eeg_events.copy()
                    new_fnirs_events = ((eeg_events_tmp[:,0]/1000)*m_1)+c_1
                    new_fnirs_events = new_fnirs_events*raw_intensity.info['sfreq']
                    eeg_events_tmp[:,0] = np.rint(new_fnirs_events)

                    reversed_event_translations = {v: k for k, v in event_translations.items()}
                    annotation = mne.annotations_from_events(eeg_events_tmp, raw_intensity.info['sfreq'], event_desc=reversed_event_translations)
                    raw_intensity = raw_intensity.set_annotations(annotation)

                    # Get crop IDs
                    crop_ids = [event_translations[i] for i in task_stimulous_to_crop[task_name]]
                    crop_index_start = np.array(np.where(np.isin(eeg_events_tmp[:,2], crop_ids))).min()
                    crop_index_end = np.array(np.where(np.isin(eeg_events_tmp[:,2], crop_ids))).max()

                    # Crop events to first - last event
                    eeg_events_tmp = eeg_events_tmp[crop_index_start:crop_index_end+1]

                    # Crop time to first-last event
                    crop_tmin = eeg_events_tmp[0][0]/raw_intensity.info['sfreq']
                    crop_tmin -= 1
                    crop_tmax = eeg_events_tmp[-1][0]/raw_intensity.info['sfreq']
                    # Adjust for tmin
                    crop_tmax += 40 + 1

                    raw_intensity = raw_intensity.crop(tmin=crop_tmin, tmax=crop_tmax, verbose=True)

                raw_dict[task_name][session_index] = raw_intensity
    # something is wrong with the concatination of the nirs vs eeg concatination

    for key in raw_dict.keys():
        print(key)
        print(raw_dict[key].keys())
    print('FNIRS')
    for task_name, sessions in eeg_translation_events_dict.items():
        print(task_name)
        session_order = list(sessions.keys())
        session_order.sort()
        print(session_order)
        for session_number in session_order:
            raw_intensity = raw_dict[task_name][session_number]
            raw_intensities.append(raw_intensity)

    raw_intensities = mne.concatenate_raws(raw_intensities)
    
    return raw_intensities

def read_subject_raw_eeg(
        root_directory, 
        tasks_to_do, 
        event_translations,
        task_stimulous_to_crop,
        eeg_coords):
    '''
    Read the raw eeg data for a subject
    input:
        root_directory (str)
        tasks_to_do (list of str)
        event_translations (dictionary of str)
        task_stimulous_to_crop (dictionary of lists of str)
        eeg_coords (dictionary of lists of float)
    '''
    events_dict = {task: {} for task in tasks_to_do}
    raw_voltages_dict = {task: {} for task in tasks_to_do}
    

    print(f'Tasks: {tasks_to_do} {root_directory}')
    for subdir, dirs, files in os.walk(root_directory):
        for task in tasks_to_do:
            for file in files:
                print(file)
                if file.startswith(task) and file.endswith('.vhdr'):
                    session_number = int(file.split(task)[1].split('.')[0])-1
                    file_path = os.path.join(subdir, file)
                    print(file_path)
                    raw_voltage = read_raw_eeg(file_path)

                    raw_voltage.annotations.rename(event_translations[task])
                    events, single_events_dict = mne.events_from_annotations(raw_voltage)

                    crop_ids = [single_events_dict[i] for i in task_stimulous_to_crop[task]]

                    # Crop to first-last event
                    crop_index_start = np.array(np.where(np.isin(events[:,2], crop_ids))).min()
                    crop_tmin = events[crop_index_start][0]/raw_voltage.info['sfreq']
                    crop_tmin -= 1
                    crop_index_end = np.array(np.where(np.isin(events[:,2], crop_ids))).max()
                    crop_tmax = events[crop_index_end][0]/raw_voltage.info['sfreq']
                    # Adjust for tmin
                    crop_tmax += 40 + 1 # crop_tmin

                    # Add in session start
                    session_start = events[crop_index_start].copy()
                    session_start[2] = int(f'2000{session_number}')
                    session_start[0] = session_start[0]

                    events_new = np.insert(events, crop_index_start, session_start, axis=0)

                    reversed_event_translations = {v: k for k, v in single_events_dict.items()}
                    reversed_event_translations[int(f'2000{session_number}')] = f'session_{session_number}'
                    annotation = mne.annotations_from_events(events_new, raw_voltage.info['sfreq'], event_desc=reversed_event_translations)
                    raw_voltage.set_annotations(annotation)
 
                    # Apply crop
                    raw_voltage.crop(tmin=crop_tmin, tmax=crop_tmax)

                    events, single_events_dict = mne.events_from_annotations(raw_voltage)
                    raw_voltages_dict[task][session_number] = raw_voltage

                    events, single_events_dict = mne.events_from_annotations(raw_voltage)
                    events_dict[task][session_number] = {}
                    events_dict[task][session_number]['events'] = events
                    events_dict[task][session_number]['translations'] = single_events_dict

    raw_voltages = []
    for task_name, sessions in events_dict.items():
        session_order = list(sessions.keys())
        session_order.sort()
        for session_number in session_order:
            raw_voltages.append(raw_voltages_dict[task_name][session_number])

    raw_eeg_voltage =  mne.concatenate_raws(raw_voltages)

    # Add locations
    locs = np.array(list(eeg_coords.values()))
    locs_dict = dict(zip(list(eeg_coords.keys()), locs))
    montage = mne.channels.make_dig_montage(locs_dict, coord_frame='unknown')
    raw_eeg_voltage.set_montage(montage)

    return raw_eeg_voltage, events_dict

def read_subjects_data(
        subjects,
        raw_data_directory,
        tasks,
        eeg_event_translations,
        nirs_event_translations,
        eeg_coords,
        tasks_stimulous_to_crop,
        trial_to_check_nirs,
        eeg_t_min=0,
        eeg_t_max=1,
        nirs_t_min=0,
        nirs_t_max=1,
        eeg_sample_rate=200,
        redo_preprocessing=False,

):
    '''Read subject eeg and nirs data
    input:
        subjects (list of str)
        eeg_root_directory (str)
        nirs_root_directory (str)
        tasks (list of str)
        eeg_event_translations (dictionary of dictionaries)
        nirs_event_translations (dictionary of dictionaries)
        eeg_coords (dictionary of lists of float)
        tasks_stimulous_to_crop (dictionary of lists of str)
        trial_to_check_nirs (dictionary of lists of str)
        eeg_t_min (float)
        eeg_t_max (float)
        nirs_t_min (float)
        nirs_t_max (float)
        eeg_sample_rate (int)
        redo_preprocessing (bool)
    '''
    processed_eeg_subject_list = []
    processed_nirs_subject_list = []

    eeg_raw_directory = os.path.join(raw_data_directory, 'eeg')
    nirs_raw_directory = os.path.join(raw_data_directory, 'nirs')
    processed_directory = os.path.join(raw_data_directory, 'processed')

    # Loop for subjects
    for subject_id in subjects:
        eeg_pickle_file_path = os.path.join(processed_directory, f'{subject_id}_eeg_processed.pkl')
        nirs_pickle_file_path = os.path.join(processed_directory, f'{subject_id}_nirs_processed.pkl')
        if (not redo_preprocessing and 
            (os.path.exists(eeg_pickle_file_path))
        ):  
            # voltage
            with open(eeg_pickle_file_path, 'rb') as file:
                eeg_processed_voltage = pickle.load(file)
        else:
            print(f'Starting eeg processing of {subject_id}')
            raw_eeg_voltage, eeg_events_dict  = read_subject_raw_eeg(
                root_directory=os.path.join(eeg_raw_directory, subject_id), 
                tasks_to_do=tasks, 
                event_translations=eeg_event_translations,
                task_stimulous_to_crop=tasks_stimulous_to_crop,
                eeg_coords=eeg_coords)
            
            eeg_processed_voltage = process_eeg_raw(raw_eeg_voltage, 
                                                    l_freq=None, 
                                                    h_freq=80, 
                                                    resample=eeg_sample_rate)
            
            print(f'eeg_before: {raw_eeg_voltage.get_data().shape}')
            print(f'eeg_after: {eeg_processed_voltage.get_data().shape}')

            with open(eeg_pickle_file_path, 'wb') as file:
                pickle.dump(eeg_processed_voltage, file, pickle.HIGHEST_PROTOCOL)

        if (not redo_preprocessing and 
            os.path.exists(nirs_pickle_file_path)
        ):
            with open(nirs_pickle_file_path, 'rb') as file:
                nirs_processed_hemoglobin = pickle.load(file)
        else:
            print(f'Starting nirs processing of {subject_id}')
            raw_nirs_intensity = read_subject_raw_nirs(
                    root_directory=os.path.join(nirs_raw_directory, subject_id),
                    tasks_to_do=tasks,
                    trial_to_check=trial_to_check_nirs[subject_id], 
                    nirs_event_translations=nirs_event_translations,
                    eeg_translation_events_dict=eeg_events_dict,
                    task_stimulous_to_crop=tasks_stimulous_to_crop)

            nirs_processed_hemoglobin = process_nirs_raw(raw_nirs_intensity, 
                                                         resample=None)

            print(f'nirs_before: {raw_nirs_intensity.get_data().shape}')
            print(f'nirs_after: {nirs_processed_hemoglobin.get_data().shape}')

            with open(nirs_pickle_file_path, 'wb') as file:
                pickle.dump(nirs_processed_hemoglobin, file, pickle.HIGHEST_PROTOCOL)

        if eeg_processed_voltage.info['sfreq'] != eeg_sample_rate:
            eeg_processed_voltage.resample(eeg_sample_rate)
        # if nirs_processed_hemoglobin.info['sfreq'] != fnirs_sample_rate:
        #     nirs_processed_hemoglobin.resample(fnirs_sample_rate)

        processed_eeg_subject_list.append(eeg_processed_voltage)
        processed_nirs_subject_list.append(nirs_processed_hemoglobin)

    # Concatenate subjects
    eeg_processed_voltage = processed_eeg_subject_list[0]
    if len(processed_eeg_subject_list) > 1:
        eeg_processed_voltage = mne.concatenate_raws(processed_eeg_subject_list, preload=False).load_data()
    nirs_processed_hemoglobin = processed_nirs_subject_list[0]
    if len(processed_nirs_subject_list) > 1:
        # preload = False because mne was throwing a format error without, load after
        nirs_processed_hemoglobin = mne.concatenate_raws(processed_nirs_subject_list, preload=False).load_data() 

    events, single_events_dict = mne.events_from_annotations(eeg_processed_voltage)
    print(f'Final eeg shape: {eeg_processed_voltage.get_data().shape}')
    print(f'\neeg events: {events.shape}')

    events, single_events_dict = mne.events_from_annotations(nirs_processed_hemoglobin)
    print(f'Final nirs shape: {nirs_processed_hemoglobin.get_data().shape}')
    print(f'\nnirs events: {events.shape}')

    return eeg_processed_voltage, nirs_processed_hemoglobin