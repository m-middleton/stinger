'''
This file contains functions to read data from the .mat files and the .vhdr files
'''

import numpy as np
from scipy.stats import linregress
import os

import mne
from scipy.io import loadmat

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

                    # Add translated eeg events
                    # check_not_same_events = np.logical_not(check_not_same_events)
                    eeg_events_tmp = eeg_events.copy()
                    new_fnirs_events = ((eeg_events_tmp[:,0]/1000)*m_1)+c_1
                    new_fnirs_events = new_fnirs_events*raw_intensity.info['sfreq']
                    eeg_events_tmp[:,0] = np.rint(new_fnirs_events)

                    reversed_event_translations = {v: k for k, v in event_translations.items()}
                    annotation = mne.annotations_from_events(eeg_events_tmp, raw_intensity.info['sfreq'], event_desc=reversed_event_translations)
                    raw_intensity.set_annotations(annotation)

                    # Crop to first - last event
                    events, single_events_dict = mne.events_from_annotations(raw_intensity)
                    crop_ids = [single_events_dict[i] for i in task_stimulous_to_crop[task_name]]
                    # crop_ids = [event_translations[i] for i in task_stimulous_to_crop[task_name] if i in event_translations.keys()]

                    # Crop to first-last event
                    crop_index_start = np.array(np.where(np.isin(events[:,2], crop_ids))).min()
                    crop_tmin = events[crop_index_start][0]/raw_intensity.info['sfreq']
                    crop_tmin -= 1
                    crop_index_end = np.array(np.where(np.isin(events[:,2], crop_ids))).max()
                    crop_tmax = events[crop_index_end][0]/raw_intensity.info['sfreq']
                    # Adjust for tmin
                    crop_tmax += 40 + 1

                    raw_intensity.crop(tmin=crop_tmin, tmax=crop_tmax)

                raw_dict[task_name][session_index] = raw_intensity
    # something is wrong with the contentation of the nirs vs eeg contentation

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