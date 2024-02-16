from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.signal as signal
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from itertools import compress

from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic, ExpSineSquared

import mne
from mne.time_frequency import tfr_morlet
from mne.channels import make_dig_montage
from mne.viz import plot_alignment, snapshot_brain_montage

def build_file_paths(subjects, tasks):
    eeg_base_path = os.path.join(BASE_PATH, EEG_PATH)
    nirs_base_path = os.path.join(BASE_PATH, NIRS_PATH)

    file_paths_dict = {}
    for subject in subjects:
        for task in tasks:
            eeg_cnt_file_path = os.path.join(eeg_base_path, subject + '-EEG', 'cnt_' + task + '.mat')
            eeg_mnt_file_path = os.path.join(eeg_base_path, subject + '-EEG', 'mnt_' + task + '.mat')
            eeg_mrk_file_path = os.path.join(eeg_base_path, subject + '-EEG', 'mrk_' + task + '.mat')

            nirs_cnt_file_path = os.path.join(nirs_base_path, subject + '-NIRS', 'cnt_' + task + '.mat')            
            nirs_mnt_file_path = os.path.join(nirs_base_path, subject + '-NIRS', 'mnt_' + task + '.mat')
            nirs_mrk_file_path = os.path.join(nirs_base_path, subject + '-NIRS', 'mrk_' + task + '.mat')
            
            file_paths_dict[f'{subject}:{task}'] = {
                                                'eeg:cnt':eeg_cnt_file_path, 
                                                'eeg:mnt':eeg_mnt_file_path, 
                                                'eeg:mrk':eeg_mrk_file_path,
                                                'nirs:cnt':nirs_cnt_file_path,
                                                'nirs:mnt':nirs_mnt_file_path,
                                                'nirs:mrk':nirs_mrk_file_path,
                                                }
            
    return file_paths_dict

def translate_labels(label_arrays):
    labels = []
    for label in label_arrays:
        labels.append(label[0])
    return labels

def eeg_mat_to_pd(cnt, mnt, mrk):
    # Process voltage data
    data_group = cnt['cnt_nback'][0][0]
    labels = translate_labels(data_group[0][0])
    data = data_group[4]

    # Process stim data
    stim_data_group = mrk['mrk_nback'][0][0]
    stim_times = stim_data_group[0].tolist()[0]

    stim_class = stim_data_group[1].T
    stim_label_names = translate_labels(stim_data_group[2][0])
    stim_indicies = np.where(stim_class == 1)[1]
    #stim_labels = [stim_label_names[i] for i in stim_indicies]

    stim_dict = dict(zip(stim_times, stim_indicies))
    df = pd.DataFrame(data, columns=labels)    
    return stim_dict, stim_label_names, df

def nirs_mat_to_pd(cnt, mnt, mrk):
    oxy_group = cnt['cnt_nback'][0][0][0]
    deoxy_group = cnt['cnt_nback'][0][0][1]

    oxy_data = oxy_group[0][0][5]
    deoxy_data = deoxy_group[0][0][5]

    stim_data_group = mrk['mrk_nback'][0][0]
    stim_times = stim_data_group[0].tolist()[0]

    stim_class = stim_data_group[1].T
    stim_label_names = translate_labels(stim_data_group[2][0])
    stim_indicies = np.where(stim_class == 1)[1]

    labels = translate_labels(oxy_group[0][0][4][0])

    snames = [f"S{idx}" for idx in SOURCE_IDS]
    dnames = [f"D{idx}" for idx in DETECTOR_IDS]
    sensor_detector_labels = [f'{m}_{n}' for m, n in zip(snames, dnames)]

    header_o = tuple(zip(labels,sensor_detector_labels,['hbo']*len(labels)))
    header_d = tuple(zip(labels,sensor_detector_labels,['hbr']*len(labels)))

    header_o = pd.MultiIndex.from_tuples(header_o, names=['channel','s_d', 'type'])
    header_d = pd.MultiIndex.from_tuples(header_d, names=['channel','s_d', 'type'])

    stim_dict = dict(zip(stim_times, stim_indicies))
    oxy_df = pd.DataFrame(oxy_data, columns=header_o)
    deoxy_df = pd.DataFrame(deoxy_data, columns=header_d)
    df = pd.concat([oxy_df,deoxy_df], axis=1)

    return stim_dict, stim_label_names, df

def read_matlab_data(file_paths_dict):
    pandas_dict = {}
    for subject_task_key, path_dict in file_paths_dict.items():
        eeg_cnt = loadmat(path_dict['eeg:cnt'])
        eeg_mnt = loadmat(path_dict['eeg:mnt'])
        eeg_mrk = loadmat(path_dict['eeg:mrk'])

        nirs_cnt = loadmat(path_dict['nirs:cnt'])
        nirs_mnt = loadmat(path_dict['nirs:mnt'])
        nirs_mrk = loadmat(path_dict['nirs:mrk'])

        eeg_stim_dict, eeg_stim_label_names, eeg_df = eeg_mat_to_pd(eeg_cnt, eeg_mnt, eeg_mrk)
        nirs_stim_dict, nirs_stim_label_names, nirs_df = nirs_mat_to_pd(nirs_cnt, nirs_mnt, nirs_mrk)

        pandas_dict[subject_task_key] = {'eeg':eeg_df,
                                         'eeg_stim':eeg_stim_dict,
                                         'eeg_stim_label_names':eeg_stim_label_names,
                                         'nirs':nirs_df,
                                         'nirs_stim':nirs_stim_dict,
                                         'nirs_stim_label_names':nirs_stim_label_names
                                         }

    return pandas_dict

def pre_process_eeg(df, low_cutoff=1, high_cutoff=45, srate=200):
    voltage = df['eeg']
    stim_dict = df['eeg_stim']
    stim_labels = df['eeg_stim_label_names']

    # unepoched, raw voltage data
    voltage = voltage.astype('float32')

    # band pass on 1-45
    b, a = signal.butter(3, [low_cutoff, high_cutoff], btype='band', fs=srate)
    # Match padding of MATLAB
    voltage = signal.filtfilt(b, a, voltage, 0, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    # normalize each channel so its mean power is 1
    voltage = voltage/voltage.mean(0)
    return voltage, stim_dict, stim_labels

def epoch_data_eeg(voltage, stim_dict, srate=200, stim_time=200):
    # Transform voltage to epoch chunks
    nt, nchan = voltage.shape
    nstim = len(stim_dict)

    stim_times = ((np.array(list(stim_dict.keys()))/1000) * srate).astype(int)
    stim_indicies = np.array(list(stim_dict.values()))

    # Epoch data
    start_time = -(int(stim_time*0.1))
    trange = np.arange(start_time, stim_time)
    stim_times[1:] += 1
    ts = (stim_times[:, np.newaxis] + trange).T

    # Copy how matlab is creating this because it is a mess
    voltage_epochs = np.reshape(voltage[ts, :],
                                (abs(stim_time)+abs(start_time), nstim, nchan),
                                order='F')  # timepoints x trials x channels
    voltage_epochs = np.transpose(voltage_epochs, [0,2,1])

    print(stim_times)

    return voltage_epochs, stim_times, stim_indicies

def epoch_data_nirs(voltage, stim_dict, srate=10, stim_time=200):
    # Transform voltage to epoch chunks
    nt, n_channels = voltage.shape
    n_markers = len(stim_dict)

    ival = [-5000, 60000]

    # si= 1000/cnt.fs;
    # TIMEEPS= si/100;
    # nMarkers= length(mrk.time);
    # len_sa= round(diff(ival)/si);
    # pos_zero= ceil((mrk.time-TIMEEPS)/si);
    # core_ival= [ceil(ival(1)/si) floor(ival(2)/si)];
    # addone= diff(core_ival)+1 < len_sa;
    # pos_end= pos_zero + floor(ival(2)/si) + addone;
    # IV= [-len_sa+1:0]'*ones(1,nMarkers) + ones(len_sa,1)*pos_end;

    si = 1000/srate
    time_eps = si/100
    len_sa = int(np.round(np.diff(ival)/si)[0])
    stim_times = np.ceil((np.array(list(stim_dict.keys())) - time_eps)/si).astype(int)
    core_ival = [np.ceil(ival[0]/si), np.floor(ival[1]/si)]
    add_one = np.diff(core_ival)+1 < len_sa
    pos_end = stim_times + np.floor(ival[1]/si) + add_one
    IV = (np.arange(-len_sa+1, 1)[:, np.newaxis] + pos_end).astype(int)


    # epo.x= reshape(cnt.x(IV, cidx), [len_sa nMarkers length(cidx)]);

    voltage_epochs = np.reshape(voltage[IV, :],
                                (len_sa, n_markers, n_channels),
                                order='F')
    voltage_epochs = np.transpose(voltage_epochs, [0,2,1])
    
    stim_indicies = np.array(list(stim_dict.values()))

    return voltage_epochs, stim_times, stim_indicies

def proc_variance(x, nSections=1, calcStd=False):
    """
    Calculate the variance in 'nSections' equally spaced intervals.
    Works for cnt and epo structures.
    
    Parameters:
    - dat: Dictionary containing data structure of continuous or epoched data.
    - nSections: Number of intervals in which variance is to be calculated.
    - calcStd: If True, standard deviation is calculated instead of variance.
    
    Returns:
    - dat: Updated data structure.
    """
    
    T, nChans, nMotos = x.shape
    inter = np.round(np.linspace(0, T, nSections+1)).astype(int)
    
    xo = np.zeros((nSections, nChans, nMotos))
    print(x.shape)
    
    times = []
    for s in range(nSections):
        Ti = np.arange(inter[s], inter[s+1])
        
        if len(Ti) == 1:
            print("Warning: calculating variance of scalar")
        
        if calcStd:
            xo[s, :, :] = np.std(x[Ti, :, :], axis=0)
        else:
            if len(Ti) == 1:
                xo[s, :, :] = x[Ti, :, :]
            else:
                if nChans * nMotos * len(Ti) <= 10**6:
                    xo[s, :, :] = np.var(x[Ti, :, :], axis=0)
                else:
                    for i in range(nMotos):
                        xo[s, :, i] = np.var(x[Ti, :, i], ddof=1, axis=0)
                        #print(xo[s, :, i])
                        #print(xo[s, :, i].shape)


        times.append(Ti[-1])
    
    return xo, times

# def matlab_percentile(x, p):
#     p = np.asarray(p, dtype=float)
#     n = len(x)
#     p = (p-50)*n/(n-1) + 50
#     p = np.clip(p, 0, 100)
#     return np.percentile(x, p, method='midpoint')

def quantile(x,q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))

def matlab_percentile(x,p):
    return(quantile(x,np.array(p)/100))

def remove_artifacts(
        voltage, 
        stim_dict,
        srate=200, 
        stim_time=200,
        whiskerperc=10,
        whiskerlength=3,
        do_bandpass=True, 
        do_silent_channels=True,
        do_unstable_channels=True,
        do_multipass=False,
        verbose=True):
    # Default options
    opt = {
        'Whiskerperc': 10,
        'Whiskerlength': 3,
        'TrialThresholdPercChannels': 0.2,
        'DoMultipass': False,
        'DoChannelMultipass': False,
        'DoRelVar': False,
        'DoUnstabChans': True,
        'DoSilentChans': True,
        'DoBandpass': True,
        'RemoveChannelsFirst': False,
        'Band': [5, 40],
        'CLab': ['not', 'E*'],
        'Visualize': False,
        'VisuLog': False,
        'Verbose': False
    }
    print('remove art')

    if do_bandpass:
        # Do 2nd time bandpass because its in matlab code
        # band pass on 1-45
        b, a = signal.butter(5, [5, 40], btype='band', fs=srate)
        # Match padding of MATLAB. channel wise. NOTE proc_channelwise calls filt over each channel. filt is maybe bad?
        voltage = signal.lfilter(b, a, voltage, axis=0)

    print(voltage[1])

    
    # TODO: Implement proc_segmentation to get epochs based on markers and intervals
    #fv = proc_segmentation(cnt, mrk, ival, CLab=opt['CLab'])
    fv, _, _ = epoch_data(voltage, stim_dict, srate=srate, stim_time=stim_time)
    print(fv.shape)
    
    # TODO: Implement proc_variance to compute variance for each trial
    fv, time = proc_variance(fv)

    V = np.squeeze(fv)
    
    nChans = fv.shape[1]
    chGood = np.arange(nChans)
    evGood = np.arange(V.shape[1])
    
    # Remove channels with low variance
    if do_silent_channels:
        rClab = np.where(np.mean(V < 0.5, axis=1) > 0.1)[0] # Not fully tested
        if len(rClab):
            V = np.delete(V, rClab, axis=0)
            chGood = np.delete(chGood, rClab)
        nfo = {'chans': [rClab]}
    
    # Calculate threshold and remove bad trials
    perc = matlab_percentile(V.ravel(), [0+whiskerperc, 100-whiskerperc]) # Not fully tested
    print(perc)

    thresh = perc[1] + whiskerlength * np.diff(perc)
    EX = (V > thresh)
    rTrials = np.where(np.mean(EX, axis=0) > opt['TrialThresholdPercChannels'])[0]
    V = np.delete(V, rTrials, axis=1)
    evGood = np.delete(evGood, rTrials)
    nfo['trials'] = [rTrials]

    # Multi-pass rejection
    goon = True
    while goon:
        perc = matlab_percentile(V.ravel(), [0+whiskerperc, 100-whiskerperc])
        print(perc)
        thresh = perc[1] + whiskerlength * np.diff(perc)
        isout = (V > thresh)
        
        rC = []
        if np.sum(isout) > 0.05 * V.shape[1]:
            qu = np.sum(isout, axis=1) / np.sum(isout)
            rC = np.where((qu > 0.1) & (np.mean(isout, axis=1) > 0.05))[0]
            V = np.delete(V, rC, axis=0)
            rClab = np.append(rClab, chGood[rC])
            nfo['chans'].append(chGood[rC])
            chGood = np.delete(chGood, rC)
        else:
            nfo['chans'].append([])

        rTr = np.where(np.any(V > thresh, axis=0))[0]
        V = np.delete(V, rTr, axis=1)
        rTrials = np.append(rTrials, evGood[rTr])
        nfo['trials'].append(evGood[rTr])
        evGood = np.delete(evGood, rTr)

        goon = do_multipass and (len(nfo['trials'][-1]) > 0 or len(nfo['chans'][-1]) > 0)
    
    # Remove unstable channels
    if do_unstable_channels:
        Vv = np.var(V, ddof=1, axis=1)
        print(Vv)        
        print(Vv.shape)
        perc = matlab_percentile(Vv, [0+whiskerperc, 100-whiskerperc])
        print(perc)
        thresh = perc[1] + whiskerlength * np.diff(perc)
        rC = np.where(Vv > thresh)[0]
        
        V = np.delete(V, rC, axis=0)
        rClab = np.append(rClab, chGood[rC])
        nfo['chans'].append(chGood[rC])
        chGood = np.delete(chGood, rC)
    #asdsad=asdasd

    #rClab= fv.clab(rClab);
    #mrk= mrk_selectEvents(mrk, 'not', rTrials);

    if verbose and len(rTrials) > 0:
        print(f'{len(rTrials)} artifact trials detected due to variance criterion. {len(evGood)} remaining')
    
    return stim_dict, rClab, rTrials, nfo

def single_epoch_process_eeg(df, srate=200, stim_time=200):
    '''EEG'''
    df['eeg'] = df['eeg'].drop(['HEOG', 'VEOG'], axis=1) # Drop EOG channels

    voltage, stim_dict, stim_labels = pre_process_eeg(df, srate=srate)
    voltage_epochs, stim_times, stim_indicies = epoch_data_eeg(voltage.copy(), stim_dict, srate=srate, stim_time=stim_time)

    # Delete artifacts
    #mrk, rClab, rTrials, nfo = remove_artifacts(voltage, stim_dict, srate=srate, stim_time=stim_time)
    #print(rTrials)
    #voltage_epochs = np.delete(voltage_epochs, rTrials, axis=2)
    #stim_times = np.delete(stim_times, rTrials, axis=0)
    #stim_indicies = np.delete(stim_indicies, rTrials, axis=0)

    # reshape array
    voltage_epochs = voltage_epochs.transpose([2,1,0])

    # Get events array index, something, id
    events = np.vstack((stim_times.astype(int), np.zeros(stim_times.size).astype(int)))
    event_array = np.vstack((events, stim_indicies.astype(int))).T
    
    channel_names = list(df['eeg'].columns)

    # Docs: https://mne.tools/stable/generated/mne.Info.html#mne.Info
    info = mne.create_info(
        channel_names,
        srate, 
        ch_types='eeg',
        verbose=None)

    # Add locations
    locs = np.array(list(EEG_COORDS.values()))/15
    locs_dict = dict(zip(list(EEG_COORDS.keys()), locs))
    montage = make_dig_montage(locs_dict, coord_frame='unknown')
    info.set_montage(montage)

    print(info)

    raw = mne.io.RawArray(voltage.T, info)

    # Docs: https://mne.tools/stable/generated/mne.EpochsArray.html
    epochs = mne.Epochs(
                raw, 
                events=event_array,
                tmin=-0.1, 
                tmax=3, 
                event_id=dict(zip(stim_labels, np.arange(len(stim_labels)))),
                baseline=None)
    
    return raw, epochs, event_array

def pre_process_nirs(df, low_cutoff=20, srate=200):
    voltage = df['nirs']
    stim_dict = df['nirs_stim']
    stim_labels = df['nirs_stim_label_names']

    # unepoched, raw voltage data
    voltage = voltage.astype('float32')

    # low pass
    b, a = signal.butter(3, 1/low_cutoff, btype='low', fs=srate)
    # Match padding of MATLAB
    voltage = signal.filtfilt(b, a, voltage, 0, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

    # normalize each channel so its mean power is 1
    #voltage = voltage/voltage.mean(0)
    return voltage, stim_dict, stim_labels

def single_epoch_process_nirs(df, 
                              srate=10, 
                              stim_time=10,    
                              t_min = -5,
                              t_max = 60):
    '''NIRS'''

    voltage, stim_dict, stim_labels = pre_process_nirs(df, srate=srate)
    voltage_epochs, stim_times, stim_indicies = epoch_data_nirs(voltage.copy(), stim_dict, srate=srate, stim_time=stim_time)
    

    #channel_types = df['nirs'].columns.get_level_values('type').tolist()

    # channel_region = list(df['nirs'].columns.get_level_values('channel').tolist())
    # channel_names = [f'{c} {t}' for c,t in zip(channel_region, channel_types)]

    # sd_names = df['nirs'].columns.get_level_values('s_d').tolist()
    # channel_names = [f'{c} {t}' for c,t in zip(sd_names, channel_types)]

    channel_names = list(df['nirs'].columns.get_level_values('channel').tolist())[:36]

    # Get events array index, something, id
    events = np.vstack((stim_times.astype(int), np.zeros(stim_times.size).astype(int)))
    event_array = np.vstack((events, stim_indicies.astype(int))).T

    print(event_array)

    print(stim_times)
    print(stim_times.shape)

    info = mne.create_info(
        ch_names=channel_names,
        ch_types='eeg', 
        sfreq=srate,
        verbose=None)
    
    locs = np.array(list(NIRS_COORDS.values()))/15
    locs_dict = dict(zip(channel_names, locs))
    montage = make_dig_montage(locs_dict, coord_frame='mri')
    info.set_montage(montage)
    

    # Docs: https://mne.tools/stable/generated/mne.Info.html#mne.Info
    # info = mne.create_info(
    #     ch_names=channel_names, 
    #     ch_types=['hbo']*36+['hbr']*36,
    #     sfreq=srate,
    #     verbose=None)
    
    # Set up digitization
    # mri_head_t, _ = mne.transforms._get_trans("fsaverage", "mri", "head",)
    # ch_locs = mne.transforms.apply_trans(mri_head_t, list(NIRS_COORDS.values()))
    # dig = mne._freesurfer.get_mni_fiducials("fsaverage", verbose=False)
    # for fid in dig:
    #     fid["r"] = mne.transforms.apply_trans(mri_head_t, fid["r"])
    #     fid["coord_frame"] = mne.io.constants.FIFF.FIFFV_COORD_HEAD
    # for ii, ch_loc in enumerate(ch_locs, 1):
    #     dig.append(
    #         dict(
    #             kind=mne.io.constants.FIFF.FIFFV_POINT_EEG,  # misnomer but probably okay
    #             r=ch_loc,
    #             ident=ii,
    #             coord_frame=mne.io.constants.FIFF.FIFFV_COORD_HEAD,
    #         )
    #     )
    # dig = mne.io._digitization._format_dig_points(dig)
    # with info._unlock():
    #     info.update(dig=dig)

    # Add locations
    # fake_s_locs = {idx.split('_')[0]: (0,0,0) for idx in sd_names}
    # fake_d_locs = {idx.split('_')[1]: (0,0,0) for idx in sd_names}

    # hbo_hbr_channel_locations = list(NIRS_COORDS.values()) + (list(NIRS_COORDS.values()))
    # locs = np.array(hbo_hbr_channel_locations)/15
    # locs_dict = dict(zip(channel_names, locs)) | fake_s_locs | fake_d_locs
    # montage = make_dig_montage(locs_dict, coord_frame='mri')
    # info.set_montage(montage)

    # hbo_hbr_channel_locations = list(NIRS_COORDS.values()) + (list(NIRS_COORDS.values()))
    # locs = np.array(hbo_hbr_channel_locations)/15
    # locs_dict = dict(zip(channel_names, locs)) | fake_s_locs | fake_d_locs
    # montage = make_dig_montage(locs_dict, coord_frame='mri')
    # info.set_montage(montage)

    print(info)

    # raw_nirs = mne.io.RawArray(voltage.T, info)
    raw_nirs_hbo = mne.io.RawArray(voltage.T[:36], info)
    raw_nirs_hbr = mne.io.RawArray(voltage.T[36:], info)

    # Docs: https://mne.tools/stable/generated/mne.EpochsArray.html
    epoch_nirs_hbo = mne.Epochs(
                raw_nirs_hbo, 
                events=event_array,
                tmin=t_min, 
                tmax=t_max, 
                event_id=dict(zip(stim_labels, np.arange(len(stim_labels)))),
                baseline=None)
    
    epoch_nirs_hbr = mne.Epochs(
                raw_nirs_hbr, 
                events=event_array,
                tmin=t_min, 
                tmax=t_max, 
                event_id=dict(zip(stim_labels, np.arange(len(stim_labels)))),
                baseline=None)
    
    return raw_nirs_hbo, raw_nirs_hbr, epoch_nirs_hbo, epoch_nirs_hbr, event_array

def single_epoch_process(df, srate_eeg=200, srate_nirs=10, stim_time=200):

    # EEG
    raw_eeg, epoch_eeg, event_array = single_epoch_process_eeg(df, srate=srate_eeg, stim_time=stim_time)
    # NIRs
    raw_nirs_hbo, raw_nirs_hbr, epoch_nirs_hbo, epoch_nirs_hbr, event_array = single_epoch_process_nirs(df, 
                                                                                                        srate=srate_nirs, 
                                                                                                        stim_time=stim_time,
                                                                                                        t_min=-5,
                                                                                                        t_max=60)
    
    return (event_array,
            mne.concatenate_raws([raw_eeg, raw_nirs]),
            mne.concatenate_epochs([epoch_eeg, epoch_nirs]) )

def main(subjects, tasks):
    tmin, tmax = -0.1, 3

    test = read_raw_eeg('/media/offset/T7_Shield/Dev/eeg_fNIRs/shin_2017/data/raw/VP001/nback3.vhdr').to_data_frame()
    root_directory = '/Users/mm/dev/super_resolution/eeg_fNIRs/shin_2017/data/raw/nirs/VP001/'
    trial_to_check = ['2016-05-26_007', '2016-05-26_008', '2016-05-26_009']
    
    raw_intensity = read_subject_raw_nirs(root_directory, trial_to_check)

    #raw_intensity.load_data()
    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )

    # Raw intensity
    raw_intensity.pick(picks[dists > 0.01])
    # raw_intensity.plot(
    #     n_channels=len(raw_intensity.ch_names), duration=500, show_scrollbars=False
    # )

    # Raw optical
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    #raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)

    # Scalp coupling analysis
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    # fig, ax = plt.subplots(layout="constrained")
    # ax.hist(sci)
    # ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
    # plt.show()

    # Mark bad channels
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

    # Plot hemo
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    #raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500, show_scrollbars=False)

    # Remove heart rate
    raw_haemo_unfiltered = raw_haemo.copy()
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    # for when, _raw in dict(Before=raw_haemo_unfiltered, After=raw_haemo).items():
    #     fig = _raw.compute_psd().plot(average=True, picks="data", exclude="bads")
    #     fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
    # plt.show()

    events, _ = mne.events_from_annotations(raw_haemo)
    # ['1.0', '13.0', '2.0', '3.0', '7.0', '8.0', '9.0']
    event_dict = {
                '0-back session': 1,
                '2-back session': 2,
                '3-back session': 3,}
    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info["sfreq"])
    #fig.subplots_adjust(right=0.7)  # make room for the legend
    # plt.show()

    # Epochs / reject / basline correction
    reject_criteria = None #dict(hbo=80e-6)
    baseline_correction = (None, 0)

    epoch_nirs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                        tmin=tmin, tmax=tmax,
                        reject=reject_criteria, reject_by_annotation=True,
                        proj=True, baseline=baseline_correction, preload=True,
                        detrend=None, verbose=True)
    #epoch_nirs.plot_drop_log()

    # input("Press Enter to continue...")
    # asdas=asdasd

    channels_eeg = []
    epoch_list_eeg = []
    raw_list_eeg = []
    for subject in subjects:
        if len(channels_eeg) == 0:
            channels_eeg = list(pandas_dict[f'{subject}:nback']['eeg'].columns)

        # EEG
        raw_eeg, epoch_eeg, event_array = single_epoch_process_eeg(pandas_dict[f'{subject}:nback'], srate=200, stim_time=200)

        epoch_list_eeg.append(epoch_eeg)
        raw_list_eeg.append(raw_eeg)

    epoch_eeg = mne.concatenate_epochs(epoch_list_eeg)
    raw_eeg = mne.concatenate_raws(raw_list_eeg)

    task_name = "3-back session"

    #picks = mne.pick_types(raw_nirs.info, meg=False, fnirs=True)
    # raw_nirs.plot(n_channels=2,
    #               duration=500, show_scrollbars=False)
    # input("Press Enter to continue...")
    # asdas=asdasd

    # picks = mne.pick_types(epoch_nirs.info, exclude=[])
    # tmin, tmax = 0, 120  # use the first 120s of data
    # fmin, fmax = 1, 5  # look at frequencies between 2 and 20Hz
    # n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
    # spectrum = raw_nirs.compute_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    # psds, freqs = spectrum.get_data(exclude=(), return_freqs=True)
    # psds = 20 * np.log10(psds)  # scale to dB
    # print(psds.shape)

    eeg_evoked = epoch_eeg[task_name].average().get_data()

    print(epoch_nirs)
    epochs_hbo = epoch_nirs.copy().pick(picks='hbo')
    epochs_hbr = epoch_nirs.copy().pick(picks='hbr')
    nirs_evoked_hbo = epochs_hbo[task_name].average(picks="hbo").get_data()
    nirs_evoked_hbr = epochs_hbr[task_name].average(picks="hbr").get_data()

    print(epoch_nirs[task_name].average().get_data().shape)
    print(nirs_evoked_hbo.shape)
    print(nirs_evoked_hbr.shape)


    all_info = mne.create_info(
        list(epoch_eeg.ch_names) + list(NIRS_COORDS.keys()),
        sfreq=10,
        ch_types='eeg',
        verbose=None)

    # Add locations
    eeg_locs = np.array(list(EEG_COORDS.values()))
    eeg_locs_dict = dict(zip(list(EEG_COORDS.keys()), eeg_locs))
    nirs_locs = np.array(list(NIRS_COORDS.values()))
    nirs_locs_dict = dict(zip(list(NIRS_COORDS.keys()), nirs_locs))

    locs_dict = eeg_locs_dict | nirs_locs_dict
    montage = make_dig_montage(locs_dict, coord_frame='unknown')
    all_info.set_montage(montage)

    def my_callback(ax, ch_idx):
        """Handle axes callback.

        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        fig = ax.get_figure()
        plt.close(fig=fig)
        if ch_idx < len(epoch_eeg.ch_names):
            epoch_eeg[task_name].plot_image(picks=epoch_eeg.ch_names[ch_idx])
        else:
            nirs_idx = ch_idx - len(epoch_eeg.ch_names)
            hbo_name = epochs_hbo.ch_names[nirs_idx] #f'{list(NIRS_COORDS.keys())[nirs_idx]}_HbO'
            hbr_name = epochs_hbr.ch_names[nirs_idx] #f'{list(NIRS_COORDS.keys())[nirs_idx]}_HbR'
            
            # Create subplot grid
            axes = dict()
            colspan = 9
            rowspan = 2
            shape = (3, 19)
            this_fig = plt.figure(layout="constrained")
            this_fig.canvas.manager.set_window_title(hbo_name)

            plt.subplot2grid(shape, (0, 0), colspan=colspan, rowspan=rowspan, fig=this_fig)
            plt.subplot2grid(shape, (2, 0), colspan=colspan, rowspan=1, fig=this_fig)

            plt.subplot2grid(shape, (0, 9), colspan=colspan, rowspan=rowspan, fig=this_fig)
            plt.subplot2grid(shape, (2, 9), colspan=colspan, rowspan=1, fig=this_fig)
            plt.subplot2grid(shape, (0, 18), colspan=1, rowspan=rowspan, fig=this_fig)

            axes[hbo_name] = this_fig.axes[:2] + [this_fig.axes[-1]]
            axes[hbr_name] = this_fig.axes[2:]

            this_fig.axes[2].axes.get_yaxis().set_visible(False)
            this_fig.axes[3].axes.get_yaxis().set_visible(False)

            epoch_nirs[task_name].plot_image(
                picks=[hbo_name,hbr_name],
                axes=axes,
                vmin=-30,
                vmax=30,
                #ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            )

    fig = all_info.get_montage().plot(kind="topomap", scale_factor=0.00001, show_names=False, show=False, sphere=0.000001)
    eeg_ax = []
    fnirs_hbo_ax = []
    fnirs_hbr_ax = []
    for ax, idx in mne.viz.iter_topography(
        all_info,
        fig=fig,
        fig_facecolor="white",
        axis_facecolor="white",
        axis_spinecolor="white",
        on_pick=my_callback,
        layout_scale=1,
        #legend=True
    ):
        if idx < len(epoch_eeg.ch_names):
            if len(eeg_ax) == 0:
                ax.plot(eeg_evoked[idx], color="green", label='EEG', linewidth=0.1)
            else:
                ax.plot(eeg_evoked[idx], color="green", linewidth=0.1)
            eeg_ax.append(ax)
        else:
            nirs_idx = idx - len(epoch_eeg.ch_names)
            if len(fnirs_hbo_ax) == 0:
                ax.plot(nirs_evoked_hbo[nirs_idx], color="red", label='fNIRs_HbO')
                ax.plot(nirs_evoked_hbr[nirs_idx], color="blue", label='fNIRs_HbR')
            else:
                ax.plot(nirs_evoked_hbo[nirs_idx], color="red")
                ax.plot(nirs_evoked_hbr[nirs_idx], color="blue")
            fnirs_hbo_ax.append(ax)
            fnirs_hbr_ax.append(ax)
        ax.set_title(f'{all_info.ch_names[idx]}',fontsize=18)

        fig.legend(loc='upper left', ncol=3)

    #plt.gcf().suptitle("Avg Evoked")
    plt.show()

    
    # spectrum = epochs[task_name].compute_psd(picks=picks, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    # psds, freqs = spectrum.get_data(exclude=(), return_freqs=True)
    # 
    # print(psds.shape)
    # aasd=asdas

    # reject_criteria = dict(
    #     eeg=0.00000001  # 3000 fT  # 3000 fT/cm
    # )  # 150 µV
    # epochs.drop_bad(reject=reject_criteria)
    # three_back_target = epochs["3-back target"].average()
    # three_back_nontarget = epochs["3-back non-target"].average()
    # three_back_session = epochs["3-back session"].average()

    #spectrum = raw.compute_psd(fmax=100)    
    #raw.plot(events=events, duration=5, n_channels=28, scalings='auto')

    #spectrum.plot(picks=channels_to_check, exclude="bads")
    #spectrum.plot_topo()

    #epochs["3-back target"].plot_image(picks=channels_to_check)
    #epochs["3-back non-target"].plot_image(picks=channels_to_check)

    # layout = mne.channels.find_layout(epochs["3-back target"].info, ch_type="eeg")
    # epochs["3-back target"].plot_topo_image(
    #     layout=layout, fig_facecolor="w", font_color="k", sigma=1, #scalings='auto'
    # )

    # evoked_diff = mne.combine_evoked([three_back_target, three_back_nontarget], weights=[1, -1])
    # evoked_diff.plot_topo(color="r", legend=False)
    # three_back_target.plot_topo(color="r", legend=False)

    input("Press Enter to continue...")
    asdas=asdasd

    #epochs["3-back target"].compute_psd().plot_topomap()
    #epochs["3-back target"].plot_psd()

    #epochs["3-back target"].compute_psd().plot(picks="eeg", exclude="bads")
    #epochs["3-back non-target"].compute_psd().plot(picks="eeg", exclude="bads")
    #epochs["3-back session"].compute_psd().plot(picks="eeg", exclude="bads")


    layout = mne.channels.find_layout(epochs["3-back target"].info, ch_type="eeg")
    epochs["3-back target"].plot_topo_image(
        layout=layout, fig_facecolor="w", font_color="k", sigma=1
    )

    epochs["3-back target"].plot_image(
        layout=layout, fig_facecolor="w", font_color="k", sigma=1
    )

    epochs["3-back target"].compute_psd().plot(picks="eeg", exclude="bads")
    epochs["3-back target"].plot_image(picks="eeg", combine="mean")
    epochs["3-back target"].plot_image(picks=channels_to_check)