import argparse
import joblib
import json
import os
import time
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

import mne
from mne import events_from_annotations

from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared

from processing.Processing_EEG import process_eeg
from processing.Processing_NIRS import process_nirs

from utilities.Plotting import plot_eeg_nirs_brain, plot_corelation_matrix
from utilities.Read_Data import read_subject_raw_nirs, read_subject_raw_eeg
from utilities.utilities import translate_channel_name_to_ch_id, find_sections, spatial_zscore

from Pipeline import start_pipeline_signal_prediction as start_pipeline

# CONSTANTS

# Paths
BASE_PATH = '/Users/mm/dev/super_resolution/eeg_fNIRs/shin_2017/data/'

ROOT_DIRECTORY_EEG = os.path.join(BASE_PATH, 'raw/eeg/')
ROOT_DIRECTORY_NIRS = os.path.join(BASE_PATH, 'raw/nirs/')

# Trial order
TRIAL_TO_CHECK_NIRS = {'VP001': {
                            'nback': ['2016-05-26_007', '2016-05-26_008', '2016-05-26_009',],
                            'gonogo': ['2016-05-26_001', '2016-05-26_002', '2016-05-26_003',],
                            'word': ['2016-05-26_004', '2016-05-26_005', '2016-05-26_006',]
                        },
                        'VP002': {
                            'nback': ['2016-05-26_016', '2016-05-26_017', '2016-05-26_018',],
                            'gonogo': ['2016-05-26_010', '2016-05-26_011', '2016-05-26_012',],
                            'word': ['2016-05-26_013', '2016-05-26_014', '2016-05-26_015',]
                        },
                        'VP003': {
                            'nback': ['2016-05-27_001', '2016-05-27_002', '2016-05-27_003',],
                            'gonogo': ['2016-05-27_007', '2016-05-27_008', '2016-05-27_009',],
                            'word': ['2016-05-27_004', '2016-05-27_005', '2016-05-27_006',]
                        },
                        'VP004': {
                            'nback': ['2016-05-30_001', '2016-05-30_002', '2016-05-30_003'],
                            'gonogo': ['2016-05-30_007', '2016-05-30_008', '2016-05-30_009'],
                            'word': ['2016-05-30_004', '2016-05-30_005', '2016-05-30_006']
                        },
                        'VP005': {
                            'nback': ['2016-05-30_010', '2016-05-30_011', '2016-05-30_012'],
                            'gonogo': ['2016-05-30_016', '2016-05-30_017', '2016-05-30_018'],
                            'word': ['2016-05-30_013', '2016-05-30_014', '2016-05-30_015']
                        },
                        'VP006': {
                            'nback': ['2016-05-31_001', '2016-05-31_002', '2016-05-31_003'],
                            'gonogo': ['2016-05-31_007', '2016-05-31_008', '2016-05-31_009'],
                            'word': ['2016-05-31_004', '2016-05-31_005', '2016-05-31_006']
                        },
                        'VP007': {
                            'nback': ['2016-06-01_001', '2016-06-01_002', '2016-06-01_003'],
                            'gonogo': ['2016-06-01_007', '2016-06-01_008', '2016-06-01_009'],
                            'word': ['2016-06-01_004', '2016-06-01_005', '2016-06-01_006']
                        },
                        'VP008': {
                            'nback': ['2016-06-02_001', '2016-06-02_002', '2016-06-02_003'],
                            'gonogo': ['2016-06-02_007', '2016-06-02_008', '2016-06-02_009'],
                            'word': ['2016-06-02_004', '2016-06-02_005', '2016-06-02_006']
                        },
                        'VP009': {
                            'nback': ['2016-06-02_010', '2016-06-02_011', '2016-06-02_012'],
                            'gonogo': ['2016-06-02_016', '2016-06-02_017', '2016-06-02_018'],
                            'word': ['2016-06-02_013', '2016-06-02_014', '2016-06-02_015']
                        },
                        'VP010': {
                            'nback': ['2016-06-03_001', '2016-06-03_002', '2016-06-03_003'],
                            'gonogo': ['2016-06-03_007', '2016-06-03_008', '2016-06-03_009'],
                            'word': ['2016-06-03_004', '2016-06-03_005', '2016-06-03_006']
                        },
                        'VP011': {
                            'nback': ['2016-06-03_010', '2016-06-03_011', '2016-06-03_012'],
                            'gonogo': ['2016-06-03_016', '2016-06-03_017', '2016-06-03_018'],
                            'word': ['2016-06-03_013', '2016-06-03_014', '2016-06-03_015']
                        },'VP012': {
                            'nback': ['2016-06-06_001', '2016-06-06_002', '2016-06-06_003'],
                            'gonogo': ['2016-06-06_007', '2016-06-06_008', '2016-06-06_009'],
                            'word': ['2016-06-06_004', '2016-06-06_005', '2016-06-06_006']
                        },'VP013': {
                            'nback': ['2016-06-06_010', '2016-06-06_011', '2016-06-06_012'],
                            'gonogo': ['2016-06-06_016', '2016-06-06_017', '2016-06-06_018'],
                            'word': ['2016-06-06_013', '2016-06-06_014', '2016-06-06_015']
                        },'VP014': {
                            'nback': ['2016-06-07_001', '2016-06-07_002', '2016-06-07_003'],
                            'gonogo': ['2016-06-07_007', '2016-06-07_008', '2016-06-07_009'],
                            'word': ['2016-06-07_004', '2016-06-07_005', '2016-06-07_006']
                        },'VP015': {
                            'nback': ['2016-06-07_010', '2016-06-07_011', '2016-06-07_012'],
                            'gonogo': ['2016-06-07_016', '2016-06-07_017', '2016-06-07_018'],
                            'word': ['2016-06-07_013', '2016-06-07_014', '2016-06-07_015']
                        },'VP016': {
                            'nback': ['2016-06-08_001', '2016-06-08_002', '2016-06-08_003'],
                            'gonogo': ['2016-06-08_007', '2016-06-08_008', '2016-06-08_009'],
                            'word': ['2016-06-08_004', '2016-06-08_005', '2016-06-08_006']
                        },'VP017': {
                            'nback': ['2016-06-09_001', '2016-06-09_002', '2016-06-09_003'],
                            'gonogo': ['2016-06-09_007', '2016-06-09_008', '2016-06-09_009'],
                            'word': ['2016-06-09_004', '2016-06-09_005', '2016-06-09_006']
                        },'VP018': {
                            'nback': ['2016-06-10_001', '2016-06-10_002', '2016-06-10_003'],
                            'gonogo': ['2016-06-10_007', '2016-06-10_008', '2016-06-10_009'],
                            'word': ['2016-06-10_004', '2016-06-10_005', '2016-06-10_006']
                        },'VP019': {
                            'nback': ['2016-06-13_001', '2016-06-13_002', '2016-06-13_003'],
                            'gonogo': ['2016-06-13_007', '2016-06-13_008', '2016-06-13_009'],
                            'word': ['2016-06-13_004', '2016-06-13_005', '2016-06-13_006']
                        },'VP020': {
                            'nback': ['2016-06-14_001', '2016-06-14_002', '2016-06-14_003'],
                            'gonogo': ['2016-06-14_007', '2016-06-14_008', '2016-06-14_009'],
                            'word': ['2016-06-14_004', '2016-06-14_005', '2016-06-14_006']
                        },'VP021': {
                            'nback': ['2016-06-14_010', '2016-06-14_011', '2016-06-14_012'],
                            'gonogo': ['2016-06-14_016', '2016-06-14_017', '2016-06-14_018'],
                            'word': ['2016-06-14_013', '2016-06-14_014', '2016-06-14_015']
                        },'VP022': {
                            'nback': ['2016-06-15_001', '2016-06-15_002', '2016-06-15_003'],
                            'gonogo': ['2016-06-15_007', '2016-06-15_008', '2016-06-15_009'],
                            'word': ['2016-06-15_004', '2016-06-15_005', '2016-06-15_006']
                        },'VP023': {
                            'nback': ['2016-06-16_001', '2016-06-16_002', '2016-06-16_003'],
                            'gonogo': ['2016-06-16_007', '2016-06-16_008', '2016-06-16_009'],
                            'word': ['2016-06-16_004', '2016-06-16_005', '2016-06-16_006']
                        },'VP024': {
                            'nback': ['2016-06-16_010', '2016-06-16_011', '2016-06-16_012'],
                            'gonogo': ['2016-06-16_016', '2016-06-16_017', '2016-06-16_018'],
                            'word': ['2016-06-16_013', '2016-06-16_014', '2016-06-16_015']
                        },
                        'VP025': {
                            'nback': ['2016-06-17_010', '2016-06-17_011', '2016-06-17_012',],
                            'gonogo': ['2016-06-17_016', '2016-06-17_017', '2016-06-17_018',],
                            'word': ['2016-06-17_013', '2016-06-17_014', '2016-06-17_015',]
                        },
                        'VP026': {
                            'nback': ['2016-07-11_001', '2016-07-11_002', '2016-07-11_003',],
                            'gonogo': ['2016-07-11_007', '2016-07-11_008', '2016-07-11_009',],
                            'word': ['2016-07-11_004', '2016-07-11_005', '2016-07-11_006',]
                        }
                    }

# Epoch size
T_MIN_EEG, T_MAX_EEG = -0.5, 0.9
T_MIN_NIRS, T_MAX_NIRS = -1, 1

# Task translation dictionaries
EEG_EVENT_TRANSLATIONS = {
            'nback': {
                'Stimulus/S 16': '0-back target',
                'Stimulus/S 48': '2-back target',
                'Stimulus/S 64': '2-back non-target',
                'Stimulus/S 80': '3-back target',
                'Stimulus/S 96': '3-back non-target',
                'Stimulus/S112': '0-back session',
                'Stimulus/S128': '2-back session',
                'Stimulus/S144': '3-back session'},
            'gonogo': {
                'Stimulus/S 16': 'go',
                'Stimulus/S 32': 'nogo',
                'Stimulus/S 48': 'gonogo session'},
            'word': {
                'Stimulus/S 16': 'verbal_fluency',
                'Stimulus/S 32': 'baseline'}
}
NIRS_EVENT_TRANSLATIONS = {
    'nback': {
        '7.0': '0-back session',
        '8.0': '2-back session',
        '9.0': '3-back session'},
    'gonogo': {
        '3.0': 'gonogo session'},
    'word': {
        '1.0': 'verbal_fluency',
        '2.0': 'baseline'}
}

# Sub tasks to crop times to for same length
TASK_STIMULOUS_TO_CROP = {'nback': ['0-back session', '2-back session', '3-back session'],
                            'gonogo': ['gonogo session'],
                            'word': ['verbal_fluency', 'baseline']
                            }

# Tasks to use for decoding
TASKS_TO_DECODE = ['0-back session', '2-back session', '3-back session']

# EEG Coordinates
EEG_COORDS = {'FP1':(-0.3090,0.9511,0.0001), #Fp1
                'AFF5':(-0.5417,0.7777,0.3163), #AFF5h
                'AFz':(0.0000,0.9230,0.3824),
                'F1':(-0.2888,0.6979,0.6542),
                'FC5':(-0.8709,0.3373,0.3549),
                'FC1':(-0.3581,0.3770,0.8532),
                'T7':(-1.0000,0.0000,0.0000),
                'C3':(-0.7066,0.0001,0.7066),
                'Cz':(0.0000,0.0002,1.0000),
                'CP5':(-0.8712,-0.3372,0.3552),
                'CP1':(-0.3580,-0.3767,0.8534),
                'P7':(-0.8090,-0.5878,-0.0001),
                'P3':(-0.5401,-0.6724,0.5045),
                'Pz':(0.0000,-0.7063,0.7065),
                'POz':(0.0000,-0.9230,0.3824),
                'O1':(-0.3090,-0.9511,0.0000),
                'FP2':(0.3091,0.9511,0.0000), #Fp2
                'AFF6':(0.5417,0.7777,0.3163), #AFF6h
                'F2':(0.2888,0.6979,0.6542),
                'FC2':(0.3581,0.3770,0.8532),
                'FC6':(0.8709,0.3373,0.3549),
                'C4':(0.7066,0.0001,0.7066),
                'T8':(1.0000,0.0000,0.0000),
                'CP2':(0.3580,-0.3767,0.8534),
                'CP6':(0.8712,-0.3372,0.3552),
                'P4':(0.5401,-0.6724,0.5045),
                'P8':(0.8090,-0.5878,-0.0001),
                'O2':(0.3090,-0.9511,0.0000),
                'TP9':(-0.8777,-0.2852,-0.3826),
                'TP10':(0.8777,-0.2853,-0.3826),
                
                'Fp1':(-0.3090,0.9511,0.0001),
                'AFF5h':(-0.5417,0.7777,0.3163),
                'Fp2':(0.3091,0.9511,0.0000),
                'AFF6h':(0.5417,0.7777,0.3163),}

# NIRS Ccoordinates
NIRS_COORDS = {
    'AF7':(-0.5878,0.809,0),
    'AFF5':(-0.6149,0.7564,0.2206),
    'AFp7':(-0.454,0.891,0),
    'AF5h':(-0.4284,0.875,0.2213),
    'AFp3':(-0.2508,0.9565,0.1438),
    'AFF3h':(-0.352,0.8111,0.4658),
    'AF1':(-0.1857,0.915,0.3558),
    'AFFz':(0,0.8312,0.5554),
    'AFpz':(0,0.9799,0.1949),
    'AF2':(0.1857,0.915,0.3558),
    'AFp4':(0.2508,0.9565,0.1437),
    'FCC3':(-0.6957,0.1838,0.6933),
    'C3h':(-0.555,0.0002,0.8306),
    'C5h':(-0.8311,0.0001,0.5552),
    'CCP3':(-0.6959,-0.1836,0.6936),
    'CPP3':(-0.6109,-0.5259,0.5904),
    'P3h':(-0.4217,-0.6869,0.5912),
    'P5h':(-0.6411,-0.6546,0.3985),
    'PPO3':(-0.4537,-0.796,0.3995),
    'AFF4h':(0.352,0.8111,0.4658),
    'AF6h':(0.4284,0.875,0.2212),
    'AFF6':(0.6149,0.7564,0.2206),
    'AFp8':(0.454,0.891,0),
    'AF8':(0.5878,0.809,0),
    'FCC4':(0.6957,0.1838,0.6933),
    'C6h':(0.8311,0.0001,0.5552),
    'C4h':(0.555,0.0002,0.8306),
    'CCP4':(0.6959,-0.1836,0.6936),
    'CPP4':(0.6109,-0.5258,0.5904),
    'P6h':(0.6411,-0.6546,0.3985),
    'P4h':(0.4216,-0.687,0.5912),
    'PPO4':(0.4537,-0.796,0.3995),
    'PPOz':(0,-0.8306,0.5551),
    'PO1':(-0.1858,-0.9151,0.3559),
    'PO2':(0.1859,-0.9151,0.3559),
    'POOz':(0,-0.9797,0.1949)}

# EEG Channels names
EEG_CHANNEL_NAMES = ['FP1', 
                    # 'AFF5h', 
                    'AFz', 
                    'F1', 
                    'FC5', 
                    'FC1', 
                    'T7', 
                    'C3', 
                    'Cz', 
                    'CP5', 
                    'CP1', 
                    'P7', 
                    'P3', 
                    'Pz', 
                    'POz', 
                    'O1',  
                    'FP2', 
                    # 'AFF6h',
                    'F2', 
                    'FC2', 
                    'FC6', 
                    'C4', 
                    'T8', 
                    'CP2', 
                    'CP6', 
                    'P4', 
                    'P8', 
                    'O2',]
    
# Fake source detector IDS
SOURCE_IDS = [1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10]
DETECTOR_IDS = list(np.arange(1,37))

# Generator function to get data
def process_sections_generator(section_indexes_nirs, 
                     section_indexes_eeg, 
                     upscale_rate, 
                     hbo_data, 
                     eeg_data, 
                     hbr_data=None,
                     do_interpolation=True):
    for section_index_nirs, section_index_eeg in zip(section_indexes_nirs, section_indexes_eeg):
        hbo_section_data = hbo_data[:, section_index_nirs[0]:section_index_nirs[1]]
        y_full = eeg_data[:, section_index_eeg[0]:section_index_eeg[1]] # eeg_section_data

        if hbr_data is not None:
            hbr_section_data = hbr_data[:, section_index_nirs[0]:section_index_nirs[1]]
            stacked_fnirs = np.vstack([hbo_section_data, hbr_section_data])
        else:
            stacked_fnirs = hbo_section_data

        # Perform time correction
        y_length = y_full.shape[1]

        # Find the nearest lower and higher multiples of upscale_rate
        lower_multiple = y_length - (y_length % upscale_rate)
        higher_multiple = lower_multiple + upscale_rate if y_length % upscale_rate != 0 else lower_multiple
        exact_multiple = upscale_rate*stacked_fnirs.shape[1]

        # Determine which multiple is closest to the original length
        y_new_length = higher_multiple
        if (y_length - lower_multiple) <= (higher_multiple - y_length):
            y_new_length = lower_multiple
        # Check if the new length is longer than the fnirs data and adjust
        if exact_multiple < y_new_length:
            y_new_length = exact_multiple
        else:
            stacked_fnirs = stacked_fnirs[:,:int(y_new_length/upscale_rate)]

        y_full = y_full[:,:y_new_length] # make eeg upscale rate times longer
        x_full = stacked_fnirs

        if do_interpolation:
            # Interpolation
            # We need to interpolate nirs to have the same number of points as eeg
            nirs_old = np.linspace(0, 1, stacked_fnirs.shape[1])  # Original sampling points for nirs
            nirs_new = np.linspace(0, 1, y_full.shape[1])  # New sampling points to match the length of eeg

            # Perform cubic interpolation
            x_full = np.zeros((stacked_fnirs.shape[0], y_full.shape[1]))
            for channel in range(stacked_fnirs.shape[0]):
                cs = CubicSpline(nirs_old, stacked_fnirs[channel])
                x_full[channel] = cs(nirs_new)
        
        # Zscore normilization
        z_sigma_max=3200
        # z_sigma_max=2200
        z_sigma_min=1

        x_full = spatial_zscore(x_full, z_sigma_max)-spatial_zscore(x_full ,z_sigma_min)
        y_full = spatial_zscore(y_full,z_sigma_max)-spatial_zscore(y_full,z_sigma_min)

        yield x_full, y_full


def get_xy_coords_signal_prediction_event_segmented(hbo_mne, 
                                    hbr_mne, 
                                    eeg_mne, 
                                    eeg_sample_rate, 
                                    fnirs_sampling_rate, 
                                    offset,
                                    nirs_dt,
                                    eeg_t_min, 
                                    eeg_t_max,
                                    tasks_to_decode,
                                    use_hbr=True):
    
    # convert index to times
    nirs_dt = int(nirs_dt*fnirs_sampling_rate)
    eeg_t_min = int(eeg_t_min*eeg_sample_rate)
    eeg_t_max = int(eeg_t_max*eeg_sample_rate)

    events_nirs, single_events_dict_nirs = mne.events_from_annotations(hbo_mne)
    events_eeg, single_events_dict_eeg = mne.events_from_annotations(eeg_mne)

    # Get session ids
    session_keys = [value for key,value in single_events_dict_nirs.items() if key.startswith('session')]
    session_keys.sort()
    print(session_keys)

    # Split processing into sessions
    section_indexes_nirs, section_indexes_eeg = find_sections(events_nirs, events_eeg, session_keys)

    # Get all data
    hbo_mne_all_data = hbo_mne.get_data()
    eeg_mne_all_data = eeg_mne.get_data()

    hbr_mne_all_data = None
    if use_hbr:
        hbr_mne_all_data = hbr_mne.get_data()

    upscale_rate = int(np.rint(eeg_sample_rate/fnirs_sampling_rate))

    x_full_list = []
    y_full_list = []
    for x_full, y_full in process_sections_generator(section_indexes_nirs, 
                                           section_indexes_eeg, 
                                           upscale_rate, 
                                           hbo_mne_all_data, 
                                           eeg_mne_all_data, 
                                           hbr_mne_all_data,
                                           do_interpolation=False):
        x_full_list.append(x_full)
        y_full_list.append(y_full)

    x_full = np.hstack(x_full_list)
    y_full = np.hstack(y_full_list)

    print(f'x_z final: {x_full.shape}')
    print(f'y_z final: {y_full.shape}')

    # Get event markers
    task_ids = [single_events_dict_eeg[task] for task in tasks_to_decode]
    events_eeg = events_eeg[np.where(np.in1d(events_eeg[:,2], task_ids))]

    y_length = y_full.shape[1]
    for index, single_event_time in enumerate(reversed(events_eeg)):
        if single_event_time[0]+nirs_dt > y_length:
            events_eeg = np.delete(events_eeg, len(events_eeg)-index-1, axis=0)
            break

    event_eeg_indexes = events_eeg[:,0]
    event_labels = events_eeg[:,2]

    event_nirs_indexes = ((events_eeg[:,0]/eeg_sample_rate)*fnirs_sampling_rate).astype('int32')

    # Segment x using list comprehension, ensuring indices stay within bounds
    segmented_x = [x_full[:,index:min(index + nirs_dt, x_full.shape[1])] for index in event_nirs_indexes]
    # Segment y with adjusted bounds handling
    segmented_y = [y_full[:,max(index + eeg_t_min, 0):min(index + eeg_t_max, y_full.shape[1])] for index in event_eeg_indexes]

    # Convert back to numpy array
    x_full = np.array(segmented_x).transpose(1,0,2)
    y_full = np.array(segmented_y).transpose(1,0,2)

    if offset != 0:
        y_full = y_full[:,:-offset]
        x_full = x_full[:,offset:]

    print(f' x End: {x_full.shape}')
    print(f' y End: {y_full.shape}')

    return x_full, y_full, event_labels


def get_xy_coords_signal_prediction_dt_segmented(hbo_mne, hbr_mne, eeg_mne, eeg_sample_rate, fnirs_sampling_rate, offset, dt, use_hbr=True):

    events_nirs, single_events_dict_nirs = mne.events_from_annotations(hbo_mne)
    events_eeg, single_events_dict_eeg = mne.events_from_annotations(eeg_mne)

    # Get session ids
    session_keys = [value for key,value in single_events_dict_nirs.items() if key.startswith('session')]
    session_keys.sort()
    print(session_keys)

    # Split processing into sessions
    section_indexes_nirs, section_indexes_eeg = find_sections(events_nirs, events_eeg, session_keys)

    # Get all data
    hbo_mne_all_data = hbo_mne.get_data()
    eeg_mne_all_data = eeg_mne.get_data()

    hbr_mne_all_data = None
    if use_hbr:
        hbr_mne_all_data = hbr_mne.get_data()

    upscale_rate = int(np.rint(eeg_sample_rate/fnirs_sampling_rate))

    x_full_list = []
    y_full_list = []
    for x_full, y_full in process_sections_generator(section_indexes_nirs, 
                                           section_indexes_eeg, 
                                           upscale_rate, 
                                           hbo_mne_all_data, 
                                           eeg_mne_all_data, 
                                           hbr_mne_all_data):
        x_full_list.append(x_full)
        y_full_list.append(y_full)

    x_full = np.hstack(x_full_list)
    y_full = np.hstack(y_full_list)

    print(f'x_z final: {x_full.shape}')
    print(f'y_z final: {y_full.shape}')

    # event_markers = np.rint((event_markers*eeg_sample_rate)/1000)
    # event_markers = list(np.intersect1d(event_markers, frame_count))

    # Get the closest multiple of dt
    y_length = y_full.shape[1]
    lower_multiple = y_length - (y_length % dt)
    higher_multiple = lower_multiple + dt if y_length % dt != 0 else lower_multiple

    # Determine which multiple is closest to the original length
    y_new_length = higher_multiple
    if (y_length - lower_multiple) <= (higher_multiple - y_length) or higher_multiple > y_length:
        y_new_length = lower_multiple

    y_full = y_full[:,:y_new_length]
    x_full = x_full[:,:y_new_length]

    print(f' x mid: {x_full.shape}')
    print(f' y mid: {y_full.shape}')

    y_full = y_full.reshape((y_full.shape[0],-1,dt))
    x_full = x_full.reshape((x_full.shape[0],-1,dt))

    if offset != 0:
        y_full = y_full[:,:-offset]
        x_full = x_full[:,offset:]

    print(f' x End: {x_full.shape}')
    print(f' y End: {y_full.shape}')

    return x_full, y_full

def get_xy_coords_signal_prediction_upscale(stacked_fnirs, eeg_voltage_data, eeg_sample_rate, fnirs_sampling_rate, offset, dt):
    y_full = eeg_voltage_data.copy()
    print(f'x Start: {stacked_fnirs.shape}')
    print(f'y Start: {y_full.shape}')

    # Perform time correction
    upscale_rate = int(np.rint(eeg_sample_rate/fnirs_sampling_rate))
    y_length = y_full.shape[0]

    # Find the nearest lower and higher multiples of upscale_rate
    lower_multiple = y_length - (y_length % upscale_rate)
    higher_multiple = lower_multiple + upscale_rate if y_length % upscale_rate != 0 else lower_multiple
    exact_multiple = upscale_rate*stacked_fnirs.shape[1]

    # Determine which multiple is closest to the original length
    y_new_length = higher_multiple
    if (y_length - lower_multiple) <= (higher_multiple - y_length):
        y_new_length = lower_multiple
    # Check if the new length is longer than the fnirs data and adjust
    if exact_multiple < y_new_length:
        y_new_length = exact_multiple
    else:
        stacked_fnirs = stacked_fnirs[:,:int(y_new_length/upscale_rate)]

    y_full = y_full[:y_new_length] # make eeg upscale rate times longer
    y_full = y_full.reshape((-1,upscale_rate)) # reshape to upscale rate times longer

    frame_count = np.array(list(range(stacked_fnirs.shape[1])))

    # y_full = y_full[frame_count-dt >= 0]
    # frame_count = frame_count[frame_count-dt >= 0]
    y_full = y_full[frame_count+dt <= stacked_fnirs.shape[1]]
    frame_count = frame_count[frame_count+dt <= stacked_fnirs.shape[1]]

    x_full = [stacked_fnirs[:, index:index+dt].flatten() for index in frame_count]
    x_full = np.array(x_full)

    ### Seperate Elements ###
    u, c = np.unique(x_full, axis=0, return_counts=True)
    assert not (c>1).any() # Check that there are no repeating input features

    # event_markers = np.rint((event_markers*eeg_sample_rate)/1000)
    # event_markers = list(np.intersect1d(event_markers, frame_count))

    print(f' x End: {x_full.shape}')
    print(f' y End: {y_full.shape}')

    return x_full, y_full

def get_xy_coords_signal_prediction_v1(stacked_fnirs, eeg_voltage_data, eeg_sample_rate, fnirs_sampling_rate, offset, dt):
    y_full = eeg_voltage_data.copy()
    
    # Perform time correction
    upscale_rate = int(np.rint(eeg_sample_rate/fnirs_sampling_rate))

    eeg_frame_count = np.array(list(range(eeg_voltage_data.shape[0])))
    eeg_frame_time = (eeg_frame_count/eeg_sample_rate)#*M_1+C_1
    fnirs_frame_count = eeg_frame_time*fnirs_sampling_rate #fnirs_events*M_2+C_2
    fnirs_frame_count = np.rint(fnirs_frame_count).astype(int)

    # dictionary mapping eeg frame to nirs frame
    fnirs_indexes = fnirs_frame_count+offset

    ### Seperate Elements ###
    # Count the occurrences of each unique element
    unique_elements, counts = np.unique(fnirs_indexes, return_counts=True)
    # Filter elements that are repeated exactly 'n' times
    elements_to_keep = unique_elements[counts == upscale_rate]
    print(fnirs_indexes[:100])
    print(counts[:100])
    print(f'elements {elements_to_keep}')
    # Create a mask to filter the original array
    mask = np.isin(fnirs_indexes, elements_to_keep)
    # Apply the mask to the array
    fnirs_indexes = fnirs_indexes[mask]
    #print(y_full.shape)
    y_full = y_full[mask]
    print(mask.shape)
    print(y_full.shape)

    y_full = y_full[fnirs_indexes-dt >= 0]
    fnirs_indexes = fnirs_indexes[fnirs_indexes-dt >= 0]
    y_full = y_full[fnirs_indexes+dt <= stacked_fnirs.shape[1]]
    fnirs_indexes = fnirs_indexes[fnirs_indexes+dt <= stacked_fnirs.shape[1]]

    print(f'x Start: {stacked_fnirs.shape}')
    print(f'y Start: {y_full.shape}')

    ### Seperate Elements ###
    fnirs_indexes, eeg_indexes =  np.unique(fnirs_indexes, return_index=True)

    x_full = [stacked_fnirs[:, index-dt:index+dt].flatten() for index in fnirs_indexes]
    x_full = np.array(x_full)

    ### Seperate Elements ###
    y_full = y_full.reshape((eeg_indexes.shape[0], -1))
    u, c = np.unique(x_full, axis=0, return_counts=True)
    if (c>1).any():
        print('WARNING: There are repeating input features')
        print(u[c>1])
        print(c[c>1])
        assert not (c>1).any() # Check that there are no repeating input features

    # event_markers = np.rint((event_markers*eeg_sample_rate)/1000)
    # event_markers = list(np.intersect1d(event_markers, fnirs_indexes))

    print(f' x End: {x_full.shape}')
    print(f' y End: {y_full.shape}')

    return x_full, y_full

def plot_train_test_acuracy(axs_train, axs_test, model, y_full, x_full, train_size, test_size, model_stats, event_markers):
    #y_mean_predict, y_std = model.predict(x_full, return_std=True)
    y_mean_predict = model.predict(x_full)
    #y_mean_predict = y_mean_predict.flatten()

    train_data = y_full[:train_size]
    train_x = x_full[:train_size]
    test_data = y_full[train_size:train_size+test_size]
    test_x = x_full[train_size:train_size+test_size]

    score_train = model.score(train_x, train_data)
    score_test = model.score(test_x, test_data)

    #train_data = train_data.flatten()
    #test_data = test_data.flatten()

    n_size_train = list(range(0, train_size))
    n_size_test = list(range(train_size, train_size+test_size))

    axs_train.plot(n_size_train, train_data, label='eeg')
    axs_test.plot(n_size_test, test_data)
    axs_train.plot(n_size_train, y_mean_predict[:train_size], label='predicted',c='orange')
    axs_test.plot(n_size_test, y_mean_predict[train_size:train_size+test_size],c='orange')
    #axs_train.errorbar(n_size_train, y_mean_predict[:train_size], y_std[:train_size], ms=0.8, alpha=0.5, fmt="o", label='predicted')
    #axs_test.errorbar(n_size_test, y_mean_predict[train_size:train_size+test_size], y_std[train_size:train_size+test_size], ms=0.8, alpha=0.5, fmt="o")

    # event_markings_train = list(np.intersect1d(event_markers, n_size_train))
    # axs_train.eventplot(event_markings_train, 'horizontal', lineoffset=-0.5, linelength=10, linestyles='--', linewidths=0.5, alpha=0.5, colors='r')
    # axs_train.set_ylim([np.min(train_data), np.max(train_data)])
    # axs_train.set_xlim([np.min(n_size_train), np.max(n_size_train)])
    
    # event_markings_test = list(np.intersect1d(event_markers, n_size_test))
    # axs_test.eventplot(event_markings_test, 'horizontal', lineoffset=-0.5, linelength=10, linestyles='--', linewidths=0.5, alpha=0.5, colors='r')
    # axs_test.set_ylim([np.min(test_data), np.max(test_data)])
    # axs_test.set_xlim([np.min(n_size_test), np.max(n_size_test)])

    if 'r2_train' in model_stats:
        axs_train.title.set_text(f"{model_stats['offset']} train: r2 {float(score_train):.2f},  psnr {float(model_stats['psnr_train']):.2f}")
    if 'r2_test' in model_stats:
        axs_test.title.set_text(f"{model_stats['offset']} test r2 {float(score_test):.2f},  psnr {float(model_stats['psnr_test']):.2f}")

def get_gpr_model():
    # kernel = 1.0 * RBF(length_scale=2, length_scale_bounds=(2, 100000.0)) + WhiteKernel(
    #     noise_level=0.316, noise_level_bounds=(0.05, 10.0)
    # )
    # kernel = 1.0 * RBF(length_scale=0.0562, length_scale_bounds=(1e-05, 100000.0)) + WhiteKernel(
    #     noise_level=0.316, noise_level_bounds=(1e-25, 10.0)
    # )
    kernel = (1.0 * 
              Matern(length_scale=0.0562, length_scale_bounds=(1e-05, 100000.0), nu=0.5) * 
              ExpSineSquared(length_scale=10.0, periodicity=1.5, length_scale_bounds=(1e-10, 1e10), periodicity_bounds=(1e-05, 200000.0)) + 
              WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-300, 10.0)
    )
    )
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=0.32, normalize_y=True)

    # print parameters
    print(gp.get_params())

    params_model = [{
        'model': [gp],
        # 'model__alpha': np.logspace(-2, 4, 5),
        'model__kernel__k1__k1__k1__constant_value_bounds': [(1e-300, 100000.0)],

        # 'model__kernel__k1__k1__k2__nu': [0.5, 1.5, 2.5, 3.5],
        # 'model__kernel__k1__k1__k2__length_scale': np.logspace(-2, 1, 5),
        # 'model__kernel__k1__k1__k2__length_scale_bounds': [(1e-10, 1e10), (1e-10, 1e15)],
        
        # 'model__kernel__k1__k2__periodicity': [0.5, 1.0, 1.5, 2.0],
        # 'model__kernel__k1__k2__length_scale': np.logspace(-2, 1, 5),
        # 'model__kernel__k1__k2__length_scale_bounds': [(1e-5, 1e5), (1e-10, 1e10)],
        # 'model__kernel__k1__k2__periodicity_bounds': [(1e-05, 150000.0), (1e-05, 200000.0)],

        # 'model__kernel__k2__noise_level': np.logspace(-2, 1, 5),
        # 'model__kernel__k2__noise_level_bounds': [(1e-300, 1e1)],

        # 'model__normalize_y': [True, False],
    }]
    return params_model

def get_random_forest_ensemble():
    
    params_model = [{
        #'model': [RandomForestRegressor(random_state=42)],
        'model__n_estimators': [10, 100, 1000],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': [1.0, 'sqrt', 'log2'],
        'model__bootstrap': [True, False],
    }]
    return params_model

def main(subjects, tasks):
    parser = argparse.ArgumentParser("Run a model over eeg and nirs data.")
    parser.add_argument("--redo_preprocessing", "--pp",
                        help="Force new preprocessing or look for existing files", 
                        default=False,
                        action='store_true')
    parser.add_argument("--model_path", "--mp", 
                        help="Filepath to model pickle file, if none will train a new model", 
                        type=str,
                        default=None)
    parser.add_argument("--plot_show", "--ps",
                        help="Show plots", 
                        default=False,
                        action='store_true')
    parser.add_argument("--plot_models", "--pm", 
                        help="Plot models", 
                        default=False,
                        action='store_true')
    parser.add_argument("--train_size", "--train", 
                        help="Samples to use in training", 
                        type=int, 
                        default=1000)
    parser.add_argument("--test_size", "--test",
                        help="Samples to use in testing", 
                        type=int, 
                        default=100)
    parser.add_argument("--eeg_sample_rate", "--e_rate",
                        help="Sampling rate to use for EEG", 
                        type=int, 
                        default=60)
    parser.add_argument("--fnirs_sample_rate", "--f_rate",
                        help="Sampling rate to use for fNIRS", 
                        type=int, 
                        default=10)
    parser.add_argument("--feature_selection", "--fs",
                        help="Number of features to use. -1 for all", 
                        type=int, 
                        default=-1)
    parser.add_argument("--model_type", "--mt",
                        help="Switch model types. 'gpr' = GPR, 'forest' = Random Forest", 
                        type=str, 
                        default='forest')
    parser.add_argument("--do_covariance", "--cv",
                        help="Plot covariance matrix", 
                        default=False,
                        action='store_true')
    parser.add_argument('--eeg_selection', 
                        '--es',
                        nargs='+', 
                        default=[])
    parser.add_argument('--nirs_selection', 
                        '--ns',
                        nargs='+', 
                        default=[])
    
    args = parser.parse_args()

    if not args.plot_show:
        plt.ioff()
    
    eeg_sample_rate = args.eeg_sample_rate
    fnirs_sample_rate = args.fnirs_sample_rate

    processed_eeg_subject_list = []
    processed_eeg_epoch_subject_list = []
    processed_nirs_subject_list = []

    # Loop for subjects
    for subject_id in subjects:
        eeg_pickle_file = os.path.join(ROOT_DIRECTORY_EEG, subject_id, f'{subject_id}_eeg_processed.pkl')
        # eeg_epochs_pickle_file = os.path.join(ROOT_DIRECTORY_EEG, subject_id, f'{subject_id}_eeg_processed_epochs.pkl')
        nirs_pickle_file = os.path.join(ROOT_DIRECTORY_NIRS, subject_id, f'{subject_id}_processed.pkl')
        if (not args.redo_preprocessing and 
            (os.path.exists(eeg_pickle_file)) #and os.path.exists(eeg_epochs_pickle_file))
        ):  
            # voltage
            with open(eeg_pickle_file, 'rb') as file:
                eeg_processed_voltage = pickle.load(file)

            # epochs
            # with open(eeg_epochs_pickle_file, 'rb') as file:
            #     epoch_eeg = pickle.load(file)
        else:
            print(f'Starting eeg processing of {subject_id}')
            raw_eeg_voltage, eeg_events_dict  = read_subject_raw_eeg(
                os.path.join(ROOT_DIRECTORY_EEG, subject_id),
                tasks,
                EEG_EVENT_TRANSLATIONS,
                TASK_STIMULOUS_TO_CROP,
                eeg_coords=EEG_COORDS)
            
            eeg_processed_voltage, epoch_eeg = process_eeg(
                raw_eeg_voltage, 
                T_MIN_EEG, 
                T_MAX_EEG,
                resample=eeg_sample_rate)
            
            print(f'eeg_before: {raw_eeg_voltage.get_data().shape}')
            print(f'eeg_after: {eeg_processed_voltage.get_data().shape}')

            with open(eeg_pickle_file, 'wb') as file:
                pickle.dump(eeg_processed_voltage, file, pickle.HIGHEST_PROTOCOL)
            # with open(eeg_epochs_pickle_file, 'wb') as file:
            #     pickle.dump(epoch_eeg, file, pickle.HIGHEST_PROTOCOL)

        if (not args.redo_preprocessing and 
            os.path.exists(nirs_pickle_file)
        ):
            with open(nirs_pickle_file, 'rb') as file:
                nirs_processed_hemoglobin = pickle.load(file)
        else:
            print(f'Starting nirs processing of {subject_id}')
            raw_nirs_intensity, raw_slope_dict = read_subject_raw_nirs(
                root_directory=os.path.join(ROOT_DIRECTORY_NIRS, subject_id),
                tasks_to_do=tasks,
                trial_to_check=TRIAL_TO_CHECK_NIRS[subject_id],
                nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
                translation_events_dict=eeg_events_dict,
                task_stimulous_to_crop=TASK_STIMULOUS_TO_CROP)

            epoch_nirs, nirs_processed_hemoglobin = process_nirs(
                raw_nirs_intensity, 
                T_MIN_NIRS, 
                T_MAX_NIRS,
                resample=None)

            print(f'nirs_before: {raw_nirs_intensity.get_data().shape}')
            print(f'nirs_after: {nirs_processed_hemoglobin.get_data().shape}')

            with open(nirs_pickle_file, 'wb') as file:
                pickle.dump(nirs_processed_hemoglobin, file, pickle.HIGHEST_PROTOCOL)

        if eeg_processed_voltage.info['sfreq'] != eeg_sample_rate:
            eeg_processed_voltage.resample(eeg_sample_rate)
        # if epoch_eeg.info['sfreq'] != eeg_sample_rate:
        #     epoch_eeg.resample(eeg_sample_rate)
        # if nirs_processed_hemoglobin.info['sfreq'] != fnirs_sample_rate:
        #     nirs_processed_hemoglobin.resample(fnirs_sample_rate)

        processed_eeg_subject_list.append(eeg_processed_voltage)
        # processed_eeg_epoch_subject_list.append(epoch_eeg)
        processed_nirs_subject_list.append(nirs_processed_hemoglobin)

    # Concatenate subjects
    eeg_processed_voltage = processed_eeg_subject_list[0]
    if len(processed_eeg_subject_list) > 1:
        eeg_processed_voltage = mne.concatenate_raws(processed_eeg_subject_list, preload=False).load_data()

    # epoch_eeg = processed_eeg_epoch_subject_list[0]
    # if len(processed_eeg_epoch_subject_list) > 1:
    #     eeg_processed_voltage = mne.concatenate_raws(processed_eeg_epoch_subject_list, preload=False).load_data()
    
    nirs_processed_hemoglobin = processed_nirs_subject_list[0]
    if len(processed_nirs_subject_list) > 1:
        # preload = False because mne was throwing a format error without, load after
        nirs_processed_hemoglobin = mne.concatenate_raws(processed_nirs_subject_list, preload=False).load_data() 

    events, single_events_dict = mne.events_from_annotations(eeg_processed_voltage)
    print(f'Final eeg shape: {eeg_processed_voltage.get_data().shape}')
    print(f'\nevents: {events.shape}')

    events, single_events_dict = mne.events_from_annotations(nirs_processed_hemoglobin)
    print(f'Final nirs shape: {nirs_processed_hemoglobin.get_data().shape}')
    print(f'\nevents: {events.shape}')
    
    # Plotting
    # eeg_processed_voltage.plot()
    # # raw_nirs_intensity.plot()
    # # input("Press Enter to continue...")
    # good_nirs = mne.pick_types(nirs_processed_hemoglobin.info, fnirs=True)
    # nirs_processed_hemoglobin.pick(picks=good_nirs).plot()
    # input("Press Enter to continue...")

    # Extract different waves from EEG
    eeg_filtered_waves = {'full': eeg_processed_voltage.copy()}
    if eeg_sample_rate/2 > 4:
        eeg_filtered_waves['delta'] = eeg_processed_voltage.copy().filter(l_freq=0.5, h_freq=4)
    if eeg_sample_rate/2 > 8:
        eeg_filtered_waves['theta'] = eeg_processed_voltage.copy().filter(l_freq=4, h_freq=8)
    if eeg_sample_rate/2 > 12:
        eeg_filtered_waves['alpha'] = eeg_processed_voltage.copy().filter(l_freq=8, h_freq=12)
    if eeg_sample_rate/2 > 30:
        eeg_filtered_waves['beta'] = eeg_processed_voltage.copy().filter(l_freq=12, h_freq=30)
    if eeg_sample_rate/2 > 80:
        eeg_filtered_waves['gamma'] = eeg_processed_voltage.copy().filter(l_freq=30, h_freq=80)

    # plot all waves
    # fig, axs = plt.subplots(len(filtered_waves_raw), 1,figsize=(50,30))
    # axs = list(axs)
    # for index, (wave_type, wave) in enumerate(filtered_waves_raw.items()):
    #     axs[index].plot(wave.get_data()[0])
    #     axs[index].set_title(f'{wave_type}')
    # fig.tight_layout()
    # pdf = PdfPages('eeg_waves_processing.pdf')
    # pdf.savefig()
    # pdf.close()
    # plt.close()
    #plt.show()
    
    # eeg_spectra = raw_eeg_voltage.compute_psd()
    # nirs_spectra = nirs_processed_hemoglobin.pick(picks=good_nirs).compute_psd()
    # fig = nirs_spectra.plot(show=False, color='red', average=True)
    # eeg_spectra.plot(axes=fig.axes[0], show=False, color='blue', average=True)
    # eeg_spectra.plot(axes=fig.axes[1], show=False, color='blue', average=True)
    # plt.show()
    # input("Press Enter to continue...")

    
    # good_eeg = mne.pick_types(eeg_filtered_waves['full'].info, eeg=True)
    # nirs_processed_hemoglobin.pick(picks=good_nirs).plot()
    # eeg_filtered_waves['full'].pick(picks=good_eeg).plot()
    # input("Press Enter to continue...")
    # asd=asasd
        
    # print(nirs_processed_hemoglobin.ch_names)
    # plot_eeg_nirs_brain(
    #     task_name, 
    #     epoch_eeg, 
    #     epoch_nirs,
    #     eeg_coords=EEG_COORDS, 
    #     nirs_cords=NIRS_COORDS)
    # input("Press Enter to continue...")
    # asdas=asdasd
    
    # Get NIRS channels to use
    if len(args.nirs_selection) == 0:
        nirs_channels_to_use_base = list(NIRS_COORDS.keys())
    else:
        nirs_channels_to_use_base = args.nirs_selection
    nirs_channels_to_use_ids = translate_channel_name_to_ch_id(NIRS_COORDS, nirs_channels_to_use_base, nirs_processed_hemoglobin.ch_names)
    print(nirs_channels_to_use_ids)

    nirs_channels_to_use_hbo = [f'{c} hbo' for c in nirs_channels_to_use_ids]
    nirs_channels_to_use_hbr = [f'{c} hbr' for c in nirs_channels_to_use_ids]
    
    # For labeling
    nirs_channels_to_use = nirs_channels_to_use_base
    # nirs_channels_to_use = [f'{c} hbo' for c in nirs_channels_to_use_ids]+[f'{c} hbr' for c in nirs_channels_to_use_ids]

    # Get EEG channels to use
    if len(args.eeg_selection) == 0:
        eeg_channels_to_use = EEG_CHANNEL_NAMES
    else:
        eeg_channels_to_use = args.eeg_selection

    # Loop through EEG wave types
    for wave_type, eeg_voltage_data_mne in eeg_filtered_waves.items():
        print(f'Wave Type: {wave_type}')
        correlation_fig,axes = plt.subplots(sharex=True, sharey = True)

        eeg_voltage_data = eeg_voltage_data_mne.copy().pick(picks=eeg_channels_to_use)
        hbo_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbo)
        hbr_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbr)

        
        #offsets = list(np.arange(-200, 200, 10))
        #offsets = [20,100,0]
        offsets = [0]
        for offset in offsets:
            if args.do_covariance:
                dt = 1000 # window size
                x_full_all_indices, y_full_all_indices = get_xy_coords_signal_prediction_dt_segmented(
                                                                            hbo_data, 
                                                                            hbr_data,
                                                                            eeg_voltage_data, 
                                                                            eeg_sample_rate, 
                                                                            fnirs_sample_rate,
                                                                            offset,
                                                                            dt,
                                                                            use_hbr=False)
            else:
                x_full_all_indices, y_full_all_indices, event_labels = get_xy_coords_signal_prediction_event_segmented(
                                                                            hbo_data, 
                                                                            hbr_data,
                                                                            eeg_voltage_data, 
                                                                            eeg_sample_rate, 
                                                                            fnirs_sample_rate,
                                                                            offset,
                                                                            nirs_dt=10,
                                                                            eeg_t_min=-1,
                                                                            eeg_t_max=1,
                                                                            tasks_to_decode=TASKS_TO_DECODE,
                                                                            use_hbr=False)
            
            
            # Loop through EEG channels
            correlation_dict_all = {}
            lag_dict_all = {}
            for eeg_id, eeg_channel in enumerate(eeg_channels_to_use):
                print(f'EEG Channel: {eeg_channel}')
                
                # Get covariance matrix
                if args.do_covariance:
                    correlation_fig, scatter, correlation_dict_single, lag_dict_single = plot_corelation_matrix(eeg_channel_name=eeg_channel,
                                        wave_type=wave_type,
                                        x_full_original=x_full_all_indices,
                                        y_full_original= y_full_all_indices[eeg_id],
                                        dt=dt,
                                        eeg_coords = EEG_COORDS,
                                        nirs_coords = NIRS_COORDS,
                                        nirs_labels=nirs_channels_to_use,
                                        subject_id=subject_id,
                                        fig=correlation_fig)
                    correlation_dict_all[eeg_channel] = correlation_dict_single
                    lag_dict_all[eeg_channel] = lag_dict_single
                # Perform EEG prediction
                else:
                    train_size = args.train_size
                    test_size = args.test_size

                    pre_load_model = False
                    if (args.model_path and
                        args.model_path.endswith('.pkl') and
                        os.path.isfile(args.model_path) ):
                        model = joblib.load(args.model_path)
                        print('Loaded model at path {args.model_path}')
                        json_path = args.model_path.replace('.pkl', '_stats.json')
                        if not os.path.exists(json_path):
                            print(f'No model stats found at path {json_path}')
                        else:
                            with open(json_path, 'r') as fp:
                                model_stats = json.load(fp)
                            print(f'Loaded model stats at path {json_path}')

                            if ('offset' in model_stats and 
                                'dt' in model_stats and 
                                'train_size' in model_stats and 
                                'test_size' in model_stats):

                                pre_load_model = True
                                offset = int(model_stats['offset'])
                                dt = int(model_stats['dt'])
                                #train_size = int(model_stats['train_size'])
                                #test_size = int(model_stats['test_size'])

                                print(model_stats)

                                # x and y from offset and dt
                                x_full, y_full, event_markers = get_xy_coords(stacked_fnirs, eeg_voltage_data.copy(), eeg_sample_rate, fnirs_sample_rate, offset, dt, epoch_eeg.events[:,0])
                                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
                                plot_train_test_acuracy(axs[0], 
                                                        axs[1], 
                                                        model, 
                                                        y_full, 
                                                        x_full, 
                                                        train_size,
                                                        test_size, 
                                                        model_stats, 
                                                        event_markers)
                                fig.tight_layout()
                                fig.legend(loc='upper left', ncol=5)
                                plt.show()

                    if not pre_load_model:

                        x_full = x_full_all_indices
                        y_full = y_full_all_indices[eeg_id]

                        # x_full_all_indices = x_full_all_indices.reshape(x_full.shape[1], -1)
                    
                        print('Training new model')
                        print(f'offset: {offset}')

                        if args.model_type == 'gpr':
                            params_model = get_gpr_model()
                        elif args.model_type == 'forest':
                            params_model = get_random_forest_ensemble()

                        save_model_path = './outputs/models/'
                        
                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        save_model_name = f'model_{args.model_type}_eeg_{eeg_channel}_dt_{dt}_offset_{offset}_tree_{timestr}'
                        model_stats = {'offset': offset, 'dt': dt, 'wave_type': wave_type, 'eeg_channel': eeg_channel, 'model': args.model_type}
                        model, model_stats = start_pipeline(
                                    x_full,
                                    y_full,
                                    params_model=params_model,
                                    column_names=nirs_channels_to_use_ids,
                                    dt=dt,
                                    train_size=train_size,
                                    test_size=test_size,
                                    features_to_use=args.feature_selection,
                                    n_search=1,
                                    save_model = True,
                                    save_model_name = save_model_name,
                                    model_path = save_model_path,
                                    model_stats=model_stats
                                    )
                        
                        print(model_stats) 
                            
                        if args.plot_models:
                            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
                            plot_train_test_acuracy(axs[0], 
                                                    axs[1], 
                                                    model, 
                                                    y_full, 
                                                    x_full, 
                                                    train_size,
                                                    test_size, 
                                                    model_stats,
                                                    None)#event_markers)
                            file_path = os.path.join(save_model_path, f'{save_model_name}.png')
                            fig.tight_layout()
                            fig.legend(loc='upper left', ncol=5)
                            fig.savefig(file_path, dpi=300)

                            if args.plot_show:
                                plt.show()
                            else:
                                plt.close()

                                        
                            # ax_train = fig.add_subplot(gs[ax_id, 0])
                            # ax_test = fig.add_subplot(gs[ax_id, 1])
                            # plot_train_test_acuracy(ax_train, 
                            #                         ax_test, 
                            #                         model, 
                            #                         y_full, 
                            #                         x_full, 
                            #                         train_size,
                            #                         test_size, 
                            #                         model_stats, 
                            #                         event_markers)
        # plotting for covariance
        if args.do_covariance:
            # Distance plot
            # Adding color bar to show the correlation values
            correlation_fig.colorbar(scatter, label='Correlation')

            # Optional: Set labels and title if desired
            axes.set_xlabel('Distance')
            axes.set_ylabel('time offset (s)')
            axes.set_title(f'{wave_type} EEG/NIRS Correlation')

            # Show the plot
            correlation_fig.savefig(os.path.join(f'{wave_type}.png'), dpi=512)
            # plt.show()
            plt.close()

            # Matrix plot
            df_correlations = pd.DataFrame(correlation_dict_all)
            df_lags = pd.DataFrame(lag_dict_all)

            df_correlations.to_csv(f'correlation_{wave_type}.csv')
            df_lags.to_csv(f'lag_{wave_type}.csv')

            # Plotting
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_correlations, annot=False, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

            # Overlaying lag values
            # We iterate over the DataFrame using `itertuples` for efficiency, but you need the indices to access elements
            for i, row in enumerate(df_correlations.index):
                for j, col in enumerate(df_correlations.columns):
                    lag_value = df_lags.loc[row, col]
                    plt.text(j+0.5, i+0.5, f'{lag_value:.0f}', 
                            horizontalalignment='center', verticalalignment='center', 
                            fontdict={'size':5, 'color':'black'})

            # Further customization
            plt.title(f'Correlation Matrix with Lag Values; {wave_type} EEG/NIRS')
            plt.xticks(ticks=np.arange(0.5, len(df_correlations.columns), 1), labels=df_correlations.columns)
            plt.yticks(ticks=np.arange(0.5, len(df_correlations.index), 1), labels=df_correlations.index)
            # Show the plot
            plt.savefig(os.path.join(f'{wave_type}_matrix.png'), dpi=512)
            # plt.show()
            plt.close()


if __name__ == "__main__":
    subject_ids = np.arange(1,27) # 1-27
    # subject_ids = [1]
    subjects = []
    for i in subject_ids:
        subjects.append(f'VP{i:03d}')
    tasks = ['nback', 'gonogo', 'word']

    main(subjects, tasks)