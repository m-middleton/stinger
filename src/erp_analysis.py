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

from processing.Processing_EEG import process_eeg_raw, process_eeg_epochs
from processing.Processing_NIRS import process_nirs_raw, process_nirs_epochs

from utilities.Plotting import plot_eeg_nirs_brain, plot_corelation_matrix
from utilities.Read_Data import read_subject_raw_nirs, read_subject_raw_eeg
from utilities.utilities import translate_channel_name_to_ch_id, find_sections, spatial_zscore

from Pipeline import start_pipeline_signal_prediction as start_pipeline
from config.Constants import *

def main(subjects, tasks):
    
    parser = argparse.ArgumentParser("Run a model over eeg and nirs data.")
    parser.add_argument("--redo_preprocessing", "--pp",
                        help="Force new preprocessing or look for existing files", 
                        default=False,
                        action='store_true')
    parser.add_argument("--eeg_sample_rate", "--e_rate",
                        help="Sampling rate to use for EEG", 
                        type=int, 
                        default=60)
    parser.add_argument("--fnirs_sample_rate", "--f_rate",
                        help="Sampling rate to use for fNIRS", 
                        type=int, 
                        default=10)
    parser.add_argument("--t_min_eeg",
                        help="min_time_eeg", 
                        type=float,
                        default=-0.5)
    parser.add_argument("--t_max_eeg",
                        help="max_time_eeg", 
                        type=float,
                        default=0.9)
    parser.add_argument("--t_min_nirs",
                        help="min_time_nirs", 
                        type=float,
                        default=-1)
    parser.add_argument("--t_max_nirs",
                        help="max_time_nirs", 
                        type=float,
                        default=5)
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
    
    eeg_sample_rate = args.eeg_sample_rate
    fnirs_sample_rate = args.fnirs_sample_rate

    processed_eeg_subject_list = []
    processed_eeg_epoch_subject_list = []
    processed_nirs_subject_list = []
    processed_nirs_epoch_subject_list = []

    # Loop for subjects
    for subject_id in subjects:
        eeg_pickle_file = os.path.join(ROOT_DIRECTORY_EEG, subject_id, f'{subject_id}_eeg_processed.pkl')
        nirs_pickle_file = os.path.join(ROOT_DIRECTORY_NIRS, subject_id, f'{subject_id}_processed.pkl')
        if (not args.redo_preprocessing and 
            os.path.exists(eeg_pickle_file)
        ):  
            # voltage
            with open(eeg_pickle_file, 'rb') as file:
                eeg_processed_voltage = pickle.load(file)
        else:
            print(f'Starting eeg processing of {subject_id}')
            raw_eeg_voltage, eeg_events_dict  = read_subject_raw_eeg(
                os.path.join(ROOT_DIRECTORY_EEG, subject_id),
                tasks,
                EEG_EVENT_TRANSLATIONS,
                TASK_STIMULOUS_TO_CROP,
                eeg_coords=EEG_COORDS)
            
            eeg_processed_voltage = process_eeg_raw(
                raw_eeg_voltage,
                l_freq=None,
                h_freq=80,
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
            raw_nirs_intensity = read_subject_raw_nirs(
                root_directory=os.path.join(ROOT_DIRECTORY_NIRS, subject_id),
                tasks_to_do=tasks,
                trial_to_check=TRIAL_TO_CHECK_NIRS[subject_id],
                nirs_event_translations=NIRS_EVENT_TRANSLATIONS,
                eeg_translation_events_dict=eeg_events_dict,
                task_stimulous_to_crop=TASK_STIMULOUS_TO_CROP)

            nirs_processed_hemoglobin = process_nirs_raw(
                raw_nirs_intensity,
                resample=None)

            print(f'nirs_before: {raw_nirs_intensity.get_data().shape}')
            print(f'nirs_after: {nirs_processed_hemoglobin.get_data().shape}')

            with open(nirs_pickle_file, 'wb') as file:
                pickle.dump(nirs_processed_hemoglobin, file, pickle.HIGHEST_PROTOCOL)
            
        if eeg_processed_voltage.info['sfreq'] != eeg_sample_rate:
            eeg_processed_voltage.resample(eeg_sample_rate)
        if nirs_processed_hemoglobin.info['sfreq'] != fnirs_sample_rate:
            nirs_processed_hemoglobin.resample(fnirs_sample_rate)

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
    print(f'Event Names: {single_events_dict}')

    events, single_events_dict = mne.events_from_annotations(nirs_processed_hemoglobin)
    print(f'Final nirs shape: {nirs_processed_hemoglobin.get_data().shape}')
    print(f'\nevents: {events.shape}')

    # Epochs
    epoch_eeg = process_eeg_epochs(eeg_processed_voltage,
                                   t_min=args.t_min_eeg, 
                                   t_max=args.t_max_eeg,
                                   baseline_correction=(None, 0))
    epoch_nirs = process_nirs_epochs(nirs_processed_hemoglobin,
                                     t_min=args.t_min_nirs, 
                                     t_max=args.t_max_nirs, 
                                     baseline_correction=None)
    
    # Plotting
    # eeg_processed_voltage.plot()
    # nirs_processed_hemoglobin.plot()
    # input("Press Enter to continue...")
    # Plot epochs
    # print epoch names

    eeg_channel_names = list(EEG_COORDS.keys())
    nirs_channel_names = list(NIRS_COORDS.keys())

    # eeg_event_3 = epoch_eeg["3-back target"].average()
    # eeg_event_2 = epoch_eeg["2-back target"].average()
    # eeg_event_30 = epoch_eeg["0-back target"].average()
    # evokeds = dict(three=eeg_event_3, two=eeg_event_2, zero=eeg_event_30)

    # eeg_event_target = epoch_eeg["3-back target"].average()
    # eeg_event_nontarget = epoch_eeg["3-back non-target"].average()
    # evokeds = dict(target=eeg_event_target, nontarget=eeg_event_nontarget)
    # mne.viz.plot_compare_evokeds(evokeds, picks=eeg_channel_names[:10], combine="mean")

    nirs_event_target = epoch_nirs["3-back target"].average()
    nirs_event_nontarget = epoch_nirs["3-back non-target"].average()
    evokeds = dict(target=nirs_event_target, nontarget=nirs_event_nontarget)
    nirs_picks = ['S2_D1 hbo', 'S2_D1 hbr']
    nirs_event_target.plot(picks=nirs_picks)

    # eeg_event.plot(gfp="only")
    # eeg_event.plot_joint()
    # eeg_event.plot(gfp=True,)

    # from nilearn.plotting import plot_design_matrix
    # import mne_nirs
    # from mne_nirs.channels import get_long_channels, get_short_channels, picks_pair_to_idx
    # from mne_nirs.experimental_design import make_first_level_design_matrix
    # from mne_nirs.statistics import run_glm

    # short_chs = get_short_channels(nirs_processed_hemoglobin)
    # nirs_processed_hemoglobin = get_long_channels(nirs_processed_hemoglobin)

    # design_matrix = make_first_level_design_matrix(
    #     nirs_processed_hemoglobin,
    #     drift_model="cosine",
    #     high_pass=0.005,  # Must be specified per experiment
    #     hrf_model="spm",
    #     stim_dur=1.0,
    # )
    # # design_matrix["ShortHbO"] = np.mean(
    # #     short_chs.copy().pick(picks="hbo").get_data(), axis=0
    # # )

    # # design_matrix["ShortHbR"] = np.mean(
    # #     short_chs.copy().pick(picks="hbr").get_data(), axis=0
    # # )
    # fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    # fig = plot_design_matrix(design_matrix, ax=ax1)
    # fig.show()
    input("Press Enter to continue...")

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


if __name__ == "__main__":
    subject_ids = np.arange(1,2) # 1-27
    # subject_ids = [1]
    subjects = []
    for i in subject_ids:
        subjects.append(f'VP{i:03d}')
    tasks = ['nback', 'gonogo', 'word']

    main(subjects, tasks)