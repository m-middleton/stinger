import argparse
import joblib
import json
import os
import time
import pickle

import numpy as np
import pandas as pd

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

from utilities.Plotting import plot_eeg_nirs_brain, plot_covariance_matrix, plot_corelation_matrix
from utilities.Read_Data import read_subject_raw_nirs, read_subject_raw_eeg
from utilities.utilities import translate_channel_name_to_ch_id

from Pipeline import start_pipeline_signal_prediction as start_pipeline

# CONSTANTS
BASE_PATH = '/Users/mm/dev/super_resolution/eeg_fNIRs/shin_2017/data/'

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

def perform_zscore_normalization(a, window_size=50):
    # Convert the arrays to Pandas Series for ease of use with rolling windows
    a_series = pd.Series(a.ravel())

    # Calculate rolling mean and standard deviation for each series
    a_rolling_mean = a_series.rolling(window=window_size).mean()
    a_rolling_std = a_series.rolling(window=window_size).std()

    # Perform sliding z-score normalization
    a_normalized = (a_series - a_rolling_mean) / a_rolling_std

    # Replace NaN values (which occur at the start of the series due to the rolling window)
    a_normalized.fillna(0, inplace=True)

    # Display the first few normalized values as an example
    #print(a_normalized.head())

    # Convert back to a numpy array
    a_normalized = a_normalized.to_numpy()

    return a_normalized

def spatial_zscore(data, s):
    start_time = time.time()
    
    # Initialize first and second moment matrices
    first_moment = np.zeros_like(data)
    second_moment = np.zeros_like(data)
    
    # Iterate over the columns of the image
    for i in range(data.shape[0]):
        first_moment[i, :] = gaussian_filter(data[i, :], s)
        second_moment[i, :] = gaussian_filter(data[i, :]**2, s)
    
    # Compute standard deviation and z-score
    data_std = np.sqrt(np.maximum(second_moment - first_moment**2, np.finfo(float).eps))
    data_z_score = (data - first_moment) / data_std
    
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    
    return data_z_score

def find_sections(events_nirs, events_eeg, markers):
    """
    Find all starting and ending indexes between two marker values.

    :param indexes: List of indexes.
    :param markers: List of marker values corresponding to each index.
    :param sections: List of tuples, each tuple contains a pair of start and end marker values.
    :return: List of tuples, each tuple contains the starting and ending indexes for each section.
    """
    section_indexes_nirs = []
    section_indexes_eeg = []
    section_start_nirs = None
    section_start_eeg = None

    for i in range(events_nirs.shape[0]):
        marker_nirs = events_nirs[i][2]
        index_nirs = events_nirs[i][0]
        index_eeg = events_eeg[i][0]

        is_session_marker = any(marker_nirs == start for start in markers)

        # Check if we have found the start of a section
        if  is_session_marker and section_start_nirs is None:
            section_start_nirs = index_nirs
            section_start_eeg = index_eeg
        # Check if we have found the end of a section
        elif is_session_marker and section_start_nirs is not None: 
            section_indexes_nirs.append((section_start_nirs, index_nirs))
            section_indexes_eeg.append((section_start_eeg, index_eeg))
            section_start_nirs = index_nirs
            section_start_eeg = index_eeg
        # Connect last section
        elif i == events_nirs.shape[0]-1 and section_start_nirs is not None:
            section_indexes_nirs.append((section_start_nirs, index_nirs))
            section_indexes_eeg.append((section_start_eeg, index_eeg))
            
    return section_indexes_nirs, section_indexes_eeg

def get_xy_coords_signal_prediction(hbo_mne, hbr_mne, eeg_mne, eeg_sample_rate, fnirs_sampling_rate, offset, dt, use_hbr=True):

    events_nirs, single_events_dict_nirs = mne.events_from_annotations(hbo_mne)
    events_eeg, single_events_dict_eeg = mne.events_from_annotations(eeg_mne)

    # Get session ids
    session_keys = [value for key,value in single_events_dict_nirs.items() if key.startswith('session')]
    session_keys.sort()
    print(session_keys)

    # Split processing into sessions
    section_indexes_nirs, section_indexes_eeg = find_sections(events_nirs, events_eeg, session_keys)

    x_full_list = []
    y_full_list = []
    for section_index_nirs, section_index_eeg in zip(section_indexes_nirs, section_indexes_eeg):
        hbo_section_data = hbo_mne.get_data()[:,section_index_nirs[0]:section_index_nirs[1]]
        eeg_section_data = eeg_mne.get_data()[:,section_index_eeg[0]:section_index_eeg[1]]

        
        if use_hbr:
            hbr_section_data = hbr_mne.get_data()[:,section_index_nirs[0]:section_index_nirs[1]]
            stacked_fnirs = stacked_fnirs = np.vstack([hbo_section_data,hbr_section_data])
        else:
            stacked_fnirs = hbo_section_data
        y_full = eeg_section_data


        print(f'x Start: {stacked_fnirs.shape}')
        print(f'y Start: {y_full.shape}')

        # Perform time correction
        upscale_rate = int(np.rint(eeg_sample_rate/fnirs_sampling_rate))
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

        # Interpolation
        # We need to interpolate nirs to have the same number of points as eeg
        nirs_old = np.linspace(0, 1, stacked_fnirs.shape[1])  # Original sampling points for nirs
        nirs_new = np.linspace(0, 1, y_full.shape[1])  # New sampling points to match the length of eeg

        # Perform cubic interpolation
        x_full = np.zeros((stacked_fnirs.shape[0], y_full.shape[1]))
        for channel in range(stacked_fnirs.shape[0]):
            cs = CubicSpline(nirs_old, stacked_fnirs[channel])
            x_full[channel] = cs(nirs_new)

        # normalize with a sliding z score
        # y_full = perform_zscore_normalization(y_full, window_size=50)
        # x_full = [perform_zscore_normalization(x_full[channel], window_size=50) for channel in range(x_full.shape[0])]
        
        # Zscore
        z_sigma_max=3200
        z_sigma_min=1

        x_full = spatial_zscore(x_full, z_sigma_max)-spatial_zscore(x_full ,z_sigma_min)
        y_full = spatial_zscore(y_full,z_sigma_max)-spatial_zscore(y_full,z_sigma_min)

        print(f'x_z: {x_full.shape}')
        print(f'y_z: {y_full.shape}')

        x_full_list.append(x_full)
        y_full_list.append(y_full)

    x_full = np.hstack(x_full_list)
    y_full = np.hstack(y_full_list)

    print(f'x_z final: {x_full.shape}')
    print(f'y_z final: {y_full.shape}')
        
    # x_full = np.array(x_full)

    ### Seperate Elements ###
    # u, c = np.unique(x_full, axis=0, return_counts=True)
    # print(u[c>1])
    # assert not (c>1).any() # Check that there are no repeating input features

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
    
    args = parser.parse_args()

    if not args.plot_show:
        plt.ioff()

    root_directory_eeg = os.path.join(BASE_PATH, 'raw/eeg/')
    root_directory_nirs = os.path.join(BASE_PATH, 'raw/nirs/')

    trial_to_check_nirs = {'VP001': {
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
    t_min_eeg, t_max_eeg = -0.5, 0.9
    t_min_nirs, t_max_nirs = -1, 1
    
    eeg_event_translations = {
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
    nirs_event_translations = {
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

    task_stimulous_to_crop = {'nback': ['0-back session', '2-back session', '3-back session'],
                                'gonogo': ['gonogo session'],
                                'word': ['verbal_fluency', 'baseline']
                                }
    
    eeg_sample_rate = args.eeg_sample_rate
    fnirs_sample_rate = args.fnirs_sample_rate

    processed_eeg_subject_list = []
    processed_nirs_subject_list = []

    for subject_id in subjects:
        eeg_pickle_file = os.path.join(root_directory_eeg, subject_id, f'{subject_id}_eeg_processed.pkl')
        nirs_pickle_file = os.path.join(root_directory_nirs, subject_id, f'{subject_id}_processed.pkl')
        if (not args.redo_preprocessing and 
            os.path.exists(eeg_pickle_file)
        ):
            with open(eeg_pickle_file, 'rb') as file:
                eeg_processed_voltage = pickle.load(file)
        else:
            print(f'Starting eeg processing of {subject_id}')
            raw_eeg_voltage, eeg_events_dict  = read_subject_raw_eeg(
                os.path.join(root_directory_eeg, subject_id),
                tasks,
                eeg_event_translations,
                task_stimulous_to_crop,
                eeg_coords=EEG_COORDS)
            
            eeg_processed_voltage, epoch_eeg = process_eeg(
                raw_eeg_voltage, 
                t_min_eeg, 
                t_max_eeg,
                resample=eeg_sample_rate)
            
            print(f'eeg_before: {raw_eeg_voltage.get_data().shape}')
            print(f'eeg_after: {eeg_processed_voltage.get_data().shape}')

            with open(eeg_pickle_file, 'wb') as file:
                pickle.dump(eeg_processed_voltage, file, pickle.HIGHEST_PROTOCOL)

        if (not args.redo_preprocessing and 
            os.path.exists(nirs_pickle_file)
        ):
            with open(nirs_pickle_file, 'rb') as file:
                nirs_processed_hemoglobin = pickle.load(file)
        else:
            print(f'Starting nirs processing of {subject_id}')
            raw_nirs_intensity, raw_slope_dict = read_subject_raw_nirs(
                root_directory=os.path.join(root_directory_nirs, subject_id),
                tasks_to_do=tasks,
                trial_to_check=trial_to_check_nirs[subject_id],
                nirs_event_translations=nirs_event_translations,
                translation_events_dict=eeg_events_dict,
                task_stimulous_to_crop=task_stimulous_to_crop)

            epoch_nirs, nirs_processed_hemoglobin = process_nirs(
                raw_nirs_intensity, 
                t_min_nirs, 
                t_max_nirs,
                resample=None)

            print(f'nirs_before: {raw_nirs_intensity.get_data().shape}')
            print(f'nirs_after: {nirs_processed_hemoglobin.get_data().shape}')

            with open(nirs_pickle_file, 'wb') as file:
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
    print(f'\nevents: {events.shape}')

    test_eeg_crops = np.array(np.where(np.isin(events[:,2], [10001,10004,10007])))
    print(events[test_eeg_crops])

    events, single_events_dict = mne.events_from_annotations(nirs_processed_hemoglobin)
    print(f'Final nirs shape: {nirs_processed_hemoglobin.get_data().shape}')
    print(f'\nevents: {events.shape}')
    test_nirs_crops = np.array(np.where(np.isin(events[:,2], [1,4,7])))
    print(events[test_nirs_crops])
    
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

    # nirs_channels_to_use = ['AF7', 
    #                         'AFF5', 
    #                         'AFp7', 
    #                         'AF5h', 
    #                         'AFF3h', 
    #                         'AFp3', 
    #                         'AF1', 
    #                         'AFpz', 
    #                         'AFFz', 
    #                         'AF2', 
    #                         'AFp4', 
    #                         'AFF4h', 
    #                         'AF6h',
    #                         'AFp8',
    #                         'AF8',
    #                         'AFF6'
    #                         ]
    nirs_channels_to_use_base = list(NIRS_COORDS.keys())
    nirs_channels_to_use_ids = translate_channel_name_to_ch_id(NIRS_COORDS, nirs_channels_to_use_base, nirs_processed_hemoglobin.ch_names)
    print(nirs_channels_to_use_ids)
    
    # print(nirs_processed_hemoglobin.ch_names)
    # plot_eeg_nirs_brain(
    #     task_name, 
    #     epoch_eeg, 
    #     epoch_nirs,
    #     eeg_coords=EEG_COORDS, 
    #     nirs_cords=NIRS_COORDS)
    # input("Press Enter to continue...")
    # asdas=asdasd


    #nirs_channels_to_use = ['S3_D3', 'S3_D2', 'S3_D4'] # 'S4_D2'
    #nirs_channels_to_use_ids = ['S1_D1','S1_D2','S2_D1','S2_D3','S3_D3', 'S3_D2', 'S3_D4','S3_D1', 'S4_D2','S4_D4']
    # nirs_channels_to_use_ids = ['S4_D2','S1_D1']
    nirs_channels_to_use_hbo = [f'{c} hbo' for c in nirs_channels_to_use_ids]
    nirs_channels_to_use_hbr = [f'{c} hbr' for c in nirs_channels_to_use_ids]
    
    nirs_channels_to_use = nirs_channels_to_use_base
    # nirs_channels_to_use = [f'{c} hbo' for c in nirs_channels_to_use_ids]+[f'{c} hbr' for c in nirs_channels_to_use_ids]

    #eeg_channels_to_use = ['AFF5','FP1', 'F1', 'AFz', 'FP2', 'F2', 'AFF6']
    # eeg_channels_to_use = ['AFz']
    eeg_channels_to_use = EEG_CHANNEL_NAMES

    dt = 200

    for wave_type, eeg_voltage_data_mne in eeg_filtered_waves.items():
        print(f'Wave Type: {wave_type}')
        correlation_fig,axes = plt.subplots(sharex=True, sharey = True)

        eeg_voltage_data = eeg_voltage_data_mne.copy().pick(picks=eeg_channels_to_use)
        hbo_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbo)
        hbr_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbr)

        x_full_all_indices, y_full_all_indices = get_xy_coords_signal_prediction(hbo_data, 
                                                                    hbr_data,
                                                                    eeg_voltage_data, 
                                                                    eeg_sample_rate, 
                                                                    fnirs_sample_rate,
                                                                    0, 
                                                                    dt,
                                                                    use_hbr=False)
        for eeg_id, eeg_channel in enumerate(eeg_channels_to_use):
            print(f'EEG Channel: {eeg_channel}')
            # eeg_channel = [eeg_channel]

            # events_from_annot_eeg, event_dict = events_from_annotations(eeg_voltage_data_mne)
            # print(f'{event_dict}')
            # print(f'{events_from_annot_eeg}')
            # eeg_translation = {112: '0-back session', 128: '2-back session', 144: '3-back session'}
            
            # events_from_annot_nirs, event_dict = events_from_annotations(nirs_processed_hemoglobin)
            # print(f'{event_dict}')
            # print(f'{events_from_annot_nirs}')
            # nirs_translations = {1: '0-back session', 2: '2-back session', 3: '3-back session'}

            # Predict EEG
            # eeg_voltage_data = eeg_voltage_data_mne.copy().pick(picks=eeg_channel)
            # hbo_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbo)
            # hbr_data = nirs_processed_hemoglobin.copy().pick(picks=nirs_channels_to_use_hbr)

            # stacked_fnirs = np.vstack([hbo_data,hbr_data])
            # stacked_fnirs = hbo_data
            # stacked_fnirs = eeg_voltage_data_mne.copy().resample(fnirs_sample_rate).pick(picks=eeg_channel).get_data()
            # eeg_voltage_data = eeg_voltage_data.T

            if True:
                # Get covariance matrix
                correlation_fig, scatter = plot_covariance_matrix(eeg_channel_name=eeg_channel,
                                    wave_type=wave_type, 
                                    x_full_original=x_full_all_indices[:36], 
                                    y_full_original= y_full_all_indices[eeg_id],
                                    dt=dt,
                                    eeg_coords = EEG_COORDS,
                                    nirs_coords = NIRS_COORDS,
                                    nirs_labels=nirs_channels_to_use,
                                    subject_id=subject_id,
                                    fig=correlation_fig)
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
                    dt = 400
                    #offsets = list(np.arange(-200, 200, 10))
                    #offsets = [20,100,0]
                    offsets = [0]

                    # if args.plot_models:
                    #     fig = plt.figure(figsize=(8, 8))
                    #     gs = fig.add_gridspec(len(offsets),2)

                    for offset in offsets:
                        # x and y from offset and dt
                        x_full, y_full_all_indexs = get_xy_coords_signal_prediction(stacked_fnirs.copy(), 
                                                                            eeg_voltage_data.copy(), 
                                                                            eeg_sample_rate, 
                                                                            fnirs_sample_rate,
                                                                            offset, 
                                                                            dt)
                        
                        
                        x_full = x_full.reshape(x_full.shape[1], -1)
                        
                        for y_index in range(y_full_all_indexs.shape[1]):
                            # y_index = 'all'
                            y_full = y_full_all_indexs[:,y_index]
                            # y_full = y_full.ravel()

                            print('Training new model')
                            print(f'offset: {offset}, prediction: {y_index}')

                            if args.model_type == 'gpr':
                                params_model = get_gpr_model()
                            elif args.model_type == 'forest':
                                params_model = get_random_forest_ensemble()

                            save_model_path = './outputs/models/'
                            
                            timestr = time.strftime("%Y%m%d-%H%M%S")
                            save_model_name = f'model_{args.model_type}_eeg_{eeg_channel}_prediction_{y_index}_dt_{dt}_offset_{offset}_tree_{timestr}'
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
        if True:
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

        #input("Press Enter to continue...")

if __name__ == "__main__":
    subject_ids = np.arange(1,3) # 1-27
    # subject_ids = [1]
    subjects = []
    for i in subject_ids:
        subjects.append(f'VP{i:03d}')
    tasks = ['nback']

    main(subjects, tasks)