import numpy as np
from scipy.stats import linregress
import os

import mne
from mne.channels import make_dig_montage

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
        translation_events_dict={},
        task_stimulous_to_crop={},
        perform_time_correction=True,
        eeg_time_max = {}):
    raw_intensities = []
    counter = 0

    raw_dict = {task: {} for task in trial_to_check.keys()}
    for subdir, dirs, files in os.walk(root_directory):
        for task_name, session_names in trial_to_check.items():
            if any(f in subdir for f in session_names) and task_name in tasks_to_do:
                print(subdir)
                raw_intensity = read_raw_nirs(subdir)

                folder_name = os.path.basename(os.path.normpath(subdir))
                if len(translation_events_dict) > 0:
                    session_index = session_names.index(folder_name)
                    eeg_events = translation_events_dict[task_name][session_index]['events'].copy()
                    event_translations = translation_events_dict[task_name][session_index]['translations'].copy()

                    # Transform events to fNIRS time
                    if perform_time_correction:
                        original_events, single_events_dict = mne.events_from_annotations(raw_intensity)
                        for v,k in single_events_dict.items():
                            if v in nirs_event_translations[task_name]:
                                original_events[:,2][original_events[:,2] == k] = event_translations[nirs_event_translations[task_name][v]]
                            else:
                                original_events = np.delete(original_events, np.where(original_events[:,2] == k), axis=0)

                        eeg_tmp = eeg_events[np.isin(eeg_events[:,2], original_events[:,2])]

                        fnirs_time = np.array(original_events[:,0])/raw_intensity.info['sfreq']
                        eeg_time = np.array(eeg_tmp[:,0])/1000

                        # Get time offset coeficients
                        lr = linregress(eeg_time, fnirs_time)
                        m_1=lr.slope
                        c_1=lr.intercept
                        
                        # Match fnirs and eeg time for the specific session by the calculated coeficients
                        # fnirs_frame_count_ind = np.array(list(range(raw_intensity.get_data().shape[1])))
                        # nirs_frame_time = fnirs_frame_count_ind/raw_intensity.info['sfreq']
                        # eeg_frame_time = (nirs_frame_time*m_1)+c_1 #(nirs_frame_time-c_1)/m_1
                        # fnirs_frame_count_sub = eeg_frame_time*raw_intensity.info['sfreq']
                        # fnirs_frame_count_sub = np.rint(fnirs_frame_count_sub).astype(int)

                        # fnirs_frame_count_sub = fnirs_frame_count_sub[fnirs_frame_count_sub < fnirs_frame_count_ind.shape[0]]
                        # fnirs_frame_count_sub = fnirs_frame_count_sub[fnirs_frame_count_sub >= 0]

                        # print(fnirs_frame_count_ind, len(fnirs_frame_count_ind))
                        # print(fnirs_frame_count_sub, len(fnirs_frame_count_sub))
                        # print(raw_intensity.get_data().shape)
                        # ad=asddas
                        
                        new_fnirs_events = ((eeg_events[:,0]/1000)*m_1)+c_1
                        new_fnirs_events = new_fnirs_events*raw_intensity.info['sfreq']
                        eeg_events[:,0] = np.rint(new_fnirs_events)
                        
                        # new_fnirs_events = (eeg_events[:,0]/1000)
                        # new_fnirs_events = new_fnirs_events*raw_intensity.info['sfreq']
                        # eeg_events[:,0] = np.rint(new_fnirs_events)

                        reversed_event_translations = {v: k for k, v in event_translations.items()}
                        annotation = mne.annotations_from_events(eeg_events, raw_intensity.info['sfreq'], event_desc=reversed_event_translations)
                        raw_intensity.set_annotations(annotation)

                    # raw_intensity.annotations.rename(event_translations[task_name])
                    # Crop to time offset of eeg
                    # raw_intensity.crop(tmin=np.min(fnirs_frame_count_sub)/raw_intensity.info['sfreq'])
                    # t_max = (eegRaw.n_times - 1) / eegRaw.info['sfreq']
                        
                    # raw_intensity.annotations.rename(nirs_event_translations[task_name])

                    # Crop to first - last event
                    events, single_events_dict = mne.events_from_annotations(raw_intensity)
                    crop_ids = [single_events_dict[i] for i in task_stimulous_to_crop[task_name]]

                    # Crop to first-last event
                    crop_index_start = np.array(np.where(np.isin(events[:,2], crop_ids))).min()
                    crop_tmin = events[crop_index_start][0]/raw_intensity.info['sfreq']
                    crop_tmin -= 1
                    crop_index_end = np.array(np.where(np.isin(events[:,2], crop_ids))).max()
                    crop_tmax = events[crop_index_end][0]/raw_intensity.info['sfreq']
                    # Adjust for tmin
                    crop_tmax += crop_tmin + 1

                    raw_intensity.crop(tmin=crop_tmin, tmax=crop_tmax)


                    events, single_events_dict = mne.events_from_annotations(raw_intensity)
                    test_nirs_crops = np.array(np.where(np.isin(events[:,2], [1,4,7])))

                    eeg_events = translation_events_dict[task_name][session_index]['events']
                    test_eeg_crops = np.array(np.where(np.isin(eeg_events[:,2], [10001,10004,10007])))
                    print(events[test_nirs_crops[0]])
                    print(test_nirs_crops)
                    print(eeg_events[test_eeg_crops[0]])
                    print(test_eeg_crops)
                    # print(f'nirs {raw_intensity.get_data().shape}')
                    # print(f'events {events[-30:]}')
                    # print(single_events_dict)
                    # asd=asdas

                raw_dict[task_name][session_index] = raw_intensity
    # something is wrong with the contentation of the nirs vs eeg contentation

    for key in raw_dict.keys():
        print(key)
        print(raw_dict[key].keys())
    print('FNIRS')
    for task_name, sessions in translation_events_dict.items():
        print(task_name)
        session_order = list(sessions.keys())
        session_order.sort()
        print(session_order)
        for session_number in session_order:
            raw_intensity = raw_dict[task_name][session_number]
            raw_intensities.append(raw_intensity)
            

    raw_intensities = mne.concatenate_raws(raw_intensities)
    return raw_intensities, {}

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
                    crop_tmax += crop_tmin + 1

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
    montage = make_dig_montage(locs_dict, coord_frame='unknown')
    raw_eeg_voltage.set_montage(montage)

    return raw_eeg_voltage, events_dict