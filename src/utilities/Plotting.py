'''
    Plotting functions for EEG and NIRS data
    - Michael M 2024
'''

import os
import multiprocessing

import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, linregress

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.pyplot import cm

import mne
from mne.channels import make_dig_montage

from scipy.spatial.distance import cdist

from utilities.Read_Data import read_matlab_file, read_subjects_data
from utilities.utilities import calculate_channel_distances, translate_channel_name_to_ch_id

from processing.Processing_EEG import process_eeg_epochs
from processing.Processing_NIRS import process_nirs_epochs

from plotting.Windowed_Correlation import rolling_correlation

from config.Constants import *

def plot_corelation_matrix(eeg_channel_name, 
                           wave_type,
                           x_full_original, 
                           y_full_original,
                           dt,
                           eeg_coords,
                           nirs_coords,
                           nirs_labels=[],
                           subject_id='',
                           plot=True,
                           fig=None):
    # Construct fake info object for locations
    all_info = mne.create_info(
        [eeg_channel_name] + nirs_labels,
        sfreq=10,
        ch_types='eeg',
        verbose=None)

    # Add locations
    eeg_locs = np.array(list(eeg_coords.values()))
    eeg_locs_dict = dict(zip(list(eeg_coords.keys()), eeg_locs))
    nirs_locs = np.array(list(nirs_coords.values()))
    nirs_locs_dict = dict(zip(list(nirs_coords.keys()), nirs_locs))

    locs_dict = eeg_locs_dict | nirs_locs_dict
    montage = make_dig_montage(locs_dict, coord_frame='unknown')
    all_info.set_montage(montage)

    print(f'x_full: {x_full_original.shape}')
    print(f'y_full: {y_full_original.shape}')

    eeg_point = np.array(list(eeg_coords[eeg_channel_name]))
    distance_dict = {}
    for nirs_name, nirs_point in nirs_coords.items():
        nirs_point = np.array(list(nirs_point))
        dist = np.linalg.norm(eeg_point-nirs_point)
        distance_dict[nirs_name] = dist

    x_list = []
    y_list = []
    correlation_list = []
    print(f'y_full: {y_full_original.shape}')

    # Split into windows
    x_dict = {}
    y_full = y_full_original.copy()
    for channel_index in range(x_full_original.shape[0]):
        name = channel_index
        if len(nirs_labels) > 0: 
            name = nirs_labels[channel_index]
        
        x_dict[name] = x_full_original[channel_index].copy()

    for x_title, x_all_time in x_dict.items():
        if x_all_time.shape[0] != y_full.shape[0]:
            print(f'{x_title} - broken shape x:{x_all_time.shape[0]} y:{y_full.shape[0]}')
            continue

        # calculate maxwindows across the time
        all_correlation, lag = mean_correlation_max(x_all_time, y_full)
        correlation_list.append(all_correlation)
        y_list.append(lag)

        x_list.append(distance_dict[x_title])

    # Create a scatter plot
    if fig is None:
        fig, ax1 = plt.subplots(sharex=True, sharey = True, figsize=(10,2))
    else:
        ax1 = fig.axes[0]

    size = np.array(correlation_list) * 30  # Adjust the factor as needed
    scatter = ax1.scatter(x_list, y_list, c=correlation_list, s=size, cmap='viridis')

    # # Adding color bar to show the correlation values
    # plt.colorbar(label='Correlation')

    # # Optional: Set labels and title if desired
    # plt.xlabel('Distance')
    # plt.ylabel('time offset')
    # plt.title('Scatter Plot with Correlation Colormap')

    correlation_dict = dict(zip(nirs_labels, correlation_list))
    lag_dict = dict(zip(nirs_labels, y_list))

    # # Show the plot
    # plt.show()
    return fig, scatter, correlation_dict, lag_dict

def calculate_correlations_max(array1, array2, size_ratio=0.5):

    # Assuming array1 and array2 are defined and have the same shape
    n = np.arange(array1.shape[0])
    n = np.random.choice(n, size=int(size_ratio * n.shape[0]), replace=True)

    # Extract the selected rows
    selected_array1 = array1[n]
    selected_array2 = array2[n]

    correlations = []
    lags = []
    for i in range(selected_array1.shape[0]):
        # single_correlations = signal.correlate(selected_array1[i,:], selected_array2[i,:], mode="full")
        x = selected_array1[i,:]
        y = selected_array2[i,:]

        single_correlations = signal.correlate(x/np.std(x), y/np.std(y), 'full') / min(len(x), len(y))
        single_lags = signal.correlation_lags(x.size, y.size, mode="full")

        correlation = single_correlations[np.argmax(single_correlations)]
        lag = single_lags[np.argmax(single_correlations)]

        correlations.append(correlation)
        lags.append(lag)

    return np.array(correlations), np.array(lags)

def mean_correlation_max(array1, array2):
    correlations, lags = calculate_correlations_max(array1, array2)

    # maximum cross correlation how much time lag
    # xcor matlab
    return np.mean(correlations), np.mean(lags)

def plot_eeg_nirs_brain(task_name, epoch_eeg, epoch_nirs, eeg_coords, nirs_cords):
    eeg_evoked = epoch_eeg[task_name].average().get_data()
    print(eeg_evoked.shape)
    
    epochs_hbo = epoch_nirs.copy().pick(picks='hbo')
    epochs_hbr = epoch_nirs.copy().pick(picks='hbr')
    nirs_evoked_hbo = epochs_hbo[task_name].average(picks="hbo").get_data()
    nirs_evoked_hbr = epochs_hbr[task_name].average(picks="hbr").get_data()

    print(epoch_nirs[task_name].average().get_data().shape)
    print(nirs_evoked_hbo.shape)
    print(nirs_evoked_hbr.shape)

    temp_epoch_eeg = epoch_eeg.drop_channels(['HEOG', 'VEOG'])

    all_info = mne.create_info(
        list(temp_epoch_eeg.ch_names) + list(nirs_cords.keys()),
        sfreq=10,
        ch_types='eeg',
        verbose=None)

    # Add locations
    eeg_locs = np.array(list(eeg_coords.values()))
    eeg_locs_dict = dict(zip(list(eeg_coords.keys()), eeg_locs))
    nirs_locs = np.array(list(nirs_cords.values()))
    nirs_locs_dict = dict(zip(list(nirs_cords.keys()), nirs_locs))

    locs_dict = eeg_locs_dict | nirs_locs_dict
    montage = make_dig_montage(locs_dict, coord_frame='unknown')
    print(epoch_eeg.ch_names)
    print(nirs_cords.keys())
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

    # nirs_idx = [0,1,2,3,4]
    # fig, axs = plt.subplots(len(nirs_idx), 1, figsize=(15, 15))

    # print(nirs_evoked_hbo.shape)
    # for ax, idx in zip(list(axs), nirs_idx):
    #     #convolved_signal = np.convolve(eeg_evoked[idx], nirs_evoked_hbo[idx], mode='full')
    #     #ax.plot(range(nirs_evoked_hbo[idx].shape[0]), nirs_evoked_hbo[idx,:], color="red", label='fNIRs_HbO')
    #     #ax.plot(range(nirs_evoked_hbr[idx].shape[0]), nirs_evoked_hbr[idx,:], color="blue", label='fNIRs_HbR')
    #     ax.plot(range(eeg_evoked[idx].shape[0]), eeg_evoked[idx,:], color="green")
    #     #ax.set_title(f'{all_info.ch_names[idx]}',fontsize=18)

    # plt.show()
    # input("Press Enter to continue...")
    # asdas=asdasd

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
                ax.plot(eeg_evoked[idx], color="green", label='EEG', linewidth=0.5)
            else:
                ax.plot(eeg_evoked[idx], color="green", linewidth=0.5)
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

def plot_correlation(nirs_channels, x_train, y_train, size):
    color = iter(cm.rainbow(np.linspace(0, 1, (len(nirs_channels)+2))))
    eeg_color = next(color)
    cor_color = next(color)

    fig= plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(7,2)

    ax_eeg = fig.add_subplot(gs[0, :])
    ax_eeg.plot(y_train, c=eeg_color, label='eeg')
    ax_eeg.title.set_text('EEG')
    ax_eeg.margins(0, 0.1)

    for i, name in enumerate(nirs_channels):
        corr = signal.correlate(y_train, x_train[:, i].reshape(size,1))
        lags = signal.correlation_lags(len(x_train[:, i]), len(y_train))
        corr /= np.max(corr)

        ax_i = i+1

        ax_orig = fig.add_subplot(gs[ax_i, 0])
        ax_corr = fig.add_subplot(gs[ax_i, 1])
        
        ax_orig.plot(x_train[:, i], c=next(color), label=f'{name}')
        if i == 0:
            ax_corr.plot(lags, corr, c=cor_color, label='Cross-corr v Lag')
        else:
            ax_corr.plot(lags, corr, c=cor_color)
        
        ax_orig.title.set_text(f'{name}')
        ax_corr.title.set_text(f'{name} Cross-corr v Lag')
        ax_orig.margins(0, 0.1)
        ax_corr.margins(0, 0.1)

    fig.tight_layout()
    #fig.legend(loc='upper left', ncol=3)
    plt.show()
    #input("Press Enter to continue...")

    n_size = list(range(len(y_train)))
    plt.plot(n_size, y_train, label='eeg')
    for i, name in enumerate(nirs_channels):
        plt.plot(n_size, x_train[:, i], label=f'nirs-{name}')

    # for i, name in enumerate(nirs_channels_to_use_hbo + nirs_channels_to_use_hbr):
    #     plt.scatter(x_train[:, i], y_train, label=f'NIRS-{name}', s=0.8)

    plt.legend()
    plt.show()

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

def plot_eeg_nirs_comparison(train_eeg_data, train_nirs_data, eeg_coords, nirs_coords, eeg_sample_rate=200, nirs_sample_rate=10.42):
    # plot 5 closest closest eeg and nirs channels on the seperate subplots

    # Find all channel pairs sorted by distance
    all_pairs = find_all_channel_pairs(eeg_coords, nirs_coords)

    # Plot the channel pairs in chunks of 10
    plot_channel_pairs(train_eeg_data, train_nirs_data, all_pairs, eeg_coords, list(nirs_coords.keys()), eeg_fs=eeg_sample_rate, nirs_fs=nirs_sample_rate)

    # Print all pairs for reference
    print("All EEG-NIRS channel pairs sorted by distance:")
    for eeg_channel, nirs_channel, distance in all_pairs:
        print(f"EEG: {eeg_channel} - NIRS: {nirs_channel} - Distance: {distance:.2f}")

def plot_scatter_between_timepoints(
    train_targets,
    train_predictions,
    test_targets,
    test_predictions,
    channels_to_use,
    ax=None
):
    if ax is None:
        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig = ax[0].figure
        ax_train, ax_test = ax

    # Determine the overall min and max for both train and test data
    all_data = np.concatenate([
        train_targets, train_predictions,
        test_targets, test_predictions
    ])
    vmin, vmax = np.min(all_data), np.max(all_data)

    for targets, predictions, label, ax in [
        (train_targets, train_predictions, 'Train', ax_train),
        (test_targets, test_predictions, 'Test', ax_test)
    ]:
        real_data = targets.flatten()
        predicted_data = predictions.flatten()

        # Create a color map from dark blue to light blue
        n_points = len(real_data)
        colors = plt.cm.Blues(np.linspace(0.3, 1, n_points))

        # Plot the scatter with color gradient
        scatter = ax.scatter(real_data, predicted_data, c=range(n_points), 
                             cmap='Blues', alpha=0.6, s=1, 
                             vmin=0, vmax=n_points-1)

        # Calculate and plot the best fit line
        slope, intercept, r_value, p_value, std_err = linregress(real_data, predicted_data)
        line = slope * np.array([vmin, vmax]) + intercept
        ax.plot([vmin, vmax], line, 'g-', label=f'Best fit (R² = {r_value**2:.3f})')

        ax.set_xlabel('Real')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{label} Scatter Plot')
        
        # Set the same limits for both axes
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

        # Add identity line
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, label='Identity')

        ax.legend()

    return fig, (ax_train, ax_test)

def plot_erp_matrix(subject_ids,
                    tasks,
                    eeg_t_min,
                    eeg_t_max,
                    nirs_t_min,
                    nirs_t_max,
                    eeg_sample_rate,
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

def plot_erp_comparison(eeg_data, 
                        mrk_data, 
                        eeg_epochs, 
                        test_events, 
                        test_channels, 
                        single_events_dict,
                        eeg_t_min,
                        eeg_t_max,
                        eeg_sample_rate):
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
            print(f"Processing ERP {single_events_dict_reverse[marker]} for {channel_name}")

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
                                  time_window = [-0.1, 0.9],
                                  sampling_rate = 200,
                                  ax=None):
    
    samples_window = [int(t * sampling_rate) for t in time_window]

    # reverse single_events_dict
    single_events_dict_reverse = {v: k for k, v in single_events_dict.items()}

    # Get unique markers for selected events
    unique_markers = [marker for marker, event in single_events_dict_reverse.items() if event in event_selection]

    # Create figure and axes if not provided
    if ax is None:
        fig, axs = plt.subplots(1, len(unique_markers), figsize=(20, 5))
    else:
        fig = ax[0].figure
        axs = ax

    # Define colors for real, predicted, and difference
    colors = ['green', 'blue', 'purple']

    # Calculate and plot ERP for each marker
    for i, marker in enumerate(unique_markers):
        # Find indices of the current marker
        marker_indices = np.where(mrk_data[:, 2] == marker)[0]
        marker_indices = mrk_data[marker_indices][:,0] # Grab sample index

        # Extract EEG data around each marker
        real_epochs = []
        predicted_epochs = []
        for idx in marker_indices:
            start = idx + samples_window[0]
            end = idx + samples_window[1]
            if start >= 0 and end < real_eeg_data.shape[1]:
                real_epochs.append(real_eeg_data[:, start:end])
                predicted_epochs.append(predicted_eeg_data[:, start:end])
        
        real_epochs = np.array(real_epochs)
        predicted_epochs = np.array(predicted_epochs)
        
        # Calculate mean and SEM
        real_mean = np.mean(real_epochs, axis=0).squeeze()
        real_sem = np.std(real_epochs, axis=0).squeeze() / np.sqrt(real_epochs.shape[0])
        pred_mean = np.mean(predicted_epochs, axis=0).squeeze()
        pred_sem = np.std(predicted_epochs, axis=0).squeeze() / np.sqrt(predicted_epochs.shape[0])
        diff_mean = real_mean - pred_mean
        diff_sem = np.sqrt(real_sem**2 + pred_sem**2)  # Propagation of uncertainty
        
        # Calculate correlation
        correlation, _ = pearsonr(real_mean, pred_mean)
        
        # Plot mean and SEM
        time = np.linspace(time_window[0], time_window[1], real_mean.shape[0])
        axs[i].plot(time, real_mean, color=colors[0], label='Real')
        axs[i].fill_between(time, real_mean - real_sem, real_mean + real_sem, color=colors[0], alpha=0.3)
        axs[i].plot(time, pred_mean, color=colors[1], label='Predicted')
        axs[i].fill_between(time, pred_mean - pred_sem, pred_mean + pred_sem, color=colors[1], alpha=0.3)
        axs[i].plot(time, diff_mean, color=colors[2], label='Difference', alpha=0.5)
        
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Amplitude')
        axs[i].axvline(x=0, color='k', linestyle='--')  # Add vertical line at t=0
        axs[i].set_title(f'Event: {single_events_dict_reverse[marker]}\nCorrelation: {correlation:.4f}')
        axs[i].legend()

    return fig, axs