import os
import multiprocessing

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.pyplot import cm

import mne
from mne.channels import make_dig_montage

def _plot_variance_corelation(wave_type, 
                           all_info,
                           data_plot_dict,
                           title='',
                           subject_id=''):
    fig = all_info.get_montage().plot(kind="topomap", scale_factor=0.00001, show_names=False, show=False, sphere=0.000001)

    # mean_val = np.mean([np.mean(cov_matrix) for cov_matrix in cov_dict.values()])
    # std_val = np.std([np.std(cov_matrix) for cov_matrix in cov_dict.values()])
    # min_val = mean_val - std_val*2
    # max_val = mean_val + std_val*2
    min_val=None
    max_val=None

    plt.axis('scaled')
    for ax, idx in mne.viz.iter_topography(
        all_info,
        fig=fig,
        fig_facecolor="white",
        axis_facecolor="white",
        axis_spinecolor="white",
        layout_scale=1,
        #legend=True
    ):
        if idx != 0:
            df = data_plot_dict[all_info.ch_names[idx]]
            ax.matshow(df)
            
            plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=45)
            plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8)
        #     sns.set_theme(font_scale=0.5)
        #     sns.heatmap(data_plot_dict[all_info.ch_names[idx]],
        #             annot=False,
        #             cbar = True,
        #             vmin=min_val, 
        #             vmax=max_val,
        #             ax = ax,
        #             fmt='.2g',
        #             cmap="YlGnBu",
        #             square=True,
        #             annot_kws={"fontsize":8},
        #             xticklabels=[],
        #             yticklabels=[]
        #             )
            ax.set_title(f"{all_info.ch_names[idx]}", fontsize=8)

    fig.legend(loc='upper left', ncol=3)
    plt.show()

    fig.suptitle(f"{title} Matrixs EEG: {all_info.ch_names[0]} Wave: {wave_type}", fontsize=14)

    
    folder_path = f'./outputs/{title}/{wave_type}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # fig.tight_layout()
    fig.savefig(os.path.join(folder_path, f'{all_info.ch_names[0]}_{subject_id}_{title}.png'), dpi=512)
    # pdf = PdfPages(f'./outputs/covariance/{eeg_channel_name}_{wave_type}_covariance.pdf')
    # pdf.savefig()
    # pdf.close()
    plt.close()

def plot_corelation_matrix(eeg_channel_name, 
                           wave_type,
                           x_full, 
                           y_full,
                           dt,
                           eeg_coords,
                           nirs_coords,
                           nirs_labels=[],
                           subject_id=''):
    
    covariance, x_dict, all_info = plot_covariance_matrix(eeg_channel_name=eeg_channel_name, 
                           wave_type=wave_type,
                           x_full=x_full, 
                           y_full=y_full,
                           dt=dt,
                           eeg_coords=eeg_coords,
                           nirs_coords=nirs_coords,
                           nirs_labels=nirs_labels,
                           subject_id=subject_id)
    cor_dict = {}
    for title, covariance in covariance.items():
        tmp = np.std(x_dict[title], axis=1)
        tmp2 = np.std(y_full, axis=1)
        std = np.mean(tmp*tmp2)
        # print(tmp.shape)
        # print(tmp2.shape)
        # print(std.shape)
        # asdas=asdsa
        cor_dict[title] = covariance/std

    _plot_variance_corelation(wave_type, all_info, cor_dict, title='correlation', subject_id=subject_id)

def plot_covariance_matrix(eeg_channel_name, 
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

    # Get the closest multiple of dt
    # y_length = y_full.shape[0]
    # lower_multiple = y_length - (y_length % dt)
    # higher_multiple = lower_multiple + dt if y_length % dt != 0 else lower_multiple

    # # Determine which multiple is closest to the original length
    # y_new_length = higher_multiple
    # if (y_length - lower_multiple) <= (higher_multiple - y_length) or higher_multiple > y_length:
    #     y_new_length = lower_multiple

    # y_full_original = y_full[:y_new_length]
    # x_full = x_full[:,:y_new_length]

    # Split into windows
    
    for offset in [0]:#np.arange(0, 30, 1):
        x_dict = {}
        # Repeat array 'y'
        # y_full = y_full_original.reshape((-1,dt))
        y_full = y_full_original.copy()
        if offset != 0:
            y_full = y_full[:-offset]

        for channel_index in range(x_full_original.shape[0]):
            name = channel_index
            if len(nirs_labels) > 0: 
                name = nirs_labels[channel_index]
            
            x_dict[name] = x_full_original[channel_index].copy()
            if offset != 0:
                x_dict[name] = x_dict[name][offset:]

        for x_title, x_all_time in x_dict.items():

            if x_all_time.shape[0] != y_full.shape[0]:
                print(f'{x_title} - broken shape x:{x_all_time.shape[0]} y:{y_full.shape[0]}')
                continue

            # calculate windows across

            # Distance
            tmp_correlation = []
            # tmp_correlation = [np.corrcoef(y_full[i, :], x_all_time[i, :])[0, 1] for i in range(x_all_time.shape[0])]

            # for i in range(x_all_time.shape[0]):
            #     tmp_correlation.append(calculate_correlation(y_full[i], x_all_time[i]))

            # for i in range(x_all_time.shape[0]):
            #     correlation = pearsonr(y_full[i,:], x_all_time[i,:])[0]
            #     tmp_correlation.append(correlation)
            # all_correlation = np.mean(tmp_correlation)
            # all_correlation = mean_correlation(y_full, x_all_time)


            print(x_all_time.shape)
            print(y_full.shape)

            mean_x_all_time = np.mean(x_all_time, axis=0)
            mean_y_full = np.mean(y_full, axis=0)

            print(mean_x_all_time.shape)
            print(mean_y_full.shape)

            correlation = signal.correlate(mean_x_all_time, mean_y_full, mode="full")
            lags = signal.correlation_lags(mean_x_all_time.size, y_full.size, mode="full")
            lag = lags[np.argmax(correlation)]
            correlation_list.append(correlation[np.argmax(correlation)])
            y_list.append(lag)

            x_list.append(distance_dict[x_title])
            # y_list.append(offset*2)
            # correlation_list.append(all_correlation)

            # pandas
            # # Build pandas
            # pandas_dict = {}
            # for i in range(x_all_time.shape[1]):
            #     pandas_dict[f'nirs_{i}'] = x_all_time[:,i]
            # for i in range(y_full.shape[1]):
            #     pandas_dict[f'eeg_{i}'] = y_full[:,i]
            # df = pd.DataFrame(pandas_dict)
            # print(df)
            # # Get covariance
            # c = df.cov()
    # Create a scatter plot
    if fig is None:
        fig, ax1 = plt.subplots(sharex=True, sharey = True, figsize=(10,2))
    else:
        ax1 = fig.axes[0]

    size = np.array(correlation_list) * 1000  # Adjust the factor as needed
    scatter = ax1.scatter(x_list, y_list, c=correlation_list, s=size, cmap='viridis')

    # # Adding color bar to show the correlation values
    # plt.colorbar(label='Correlation')

    # # Optional: Set labels and title if desired
    # plt.xlabel('Distance')
    # plt.ylabel('time offset')
    # plt.title('Scatter Plot with Correlation Colormap')

    # # Show the plot
    # plt.show()
    return fig, scatter

def calculate_correlations(array1, array2, size_ratio=0.5):

    # Assuming array1 and array2 are defined and have the same shape
    n = np.arange(array1.shape[0])
    n = np.random.choice(n, size=int(size_ratio * n.shape[0]), replace=True)

    # Extract the selected rows
    selected_array1 = array1[n]
    selected_array2 = array2[n]

    # Standardize each array
    standardized_array1 = (selected_array1 - selected_array1.mean(axis=1, keepdims=True)) / selected_array1.std(axis=1, keepdims=True)
    standardized_array2 = (selected_array2 - selected_array2.mean(axis=1, keepdims=True)) / selected_array2.std(axis=1, keepdims=True)

    # Compute the correlation coefficients
    correlations = (standardized_array1 * standardized_array2).mean(axis=1)

    return correlations

def mean_correlation(array1, array2):
    correlations = calculate_correlations(array1, array2)
    test = np.nanmean(correlations)
    if np.any(np.isnan([test])):
        print(correlations)
        print(test)
        print(array1)
        print(array2)
        print(array1.shape)
        print(array2.shape)
        asd=asdsds

    # maximum cross correlation how much time lag
    # xcor matlab
    return np.nanmean(correlations)  # nanmean to handle NaN values safely

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


'''
def plot_covariance_matrix(eeg_channel_name, 
                           wave_type,
                           x_full, 
                           y_full,
                           dt,
                           eeg_coords,
                           nirs_coords,
                           nirs_labels=[],
                           subject_id='',
                           plot=True):
    # Construct info object
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

    print(f'x_full: {x_full.shape}')
    print(f'y_full: {y_full.shape}')


    eeg_point = eeg_coords[eeg_channel_name]
    distance_dict = {}
    for nirs_name, nirs_point in nirs_locs.items():
        dist = np.linalg.norm(eeg_point-nirs_point)
        distance_dict[nirs_name] = dist

    # Repeat array 'y'
    if len(nirs_labels) > 0: 
        x_dict = {nirs_labels[ind]:x_full[:,window_start:window_start+(dt*2)] for ind, window_start in enumerate(list(range(0,x_full.shape[1],dt*2)))}
    else:
        x_dict = {ind:x_full[:,window_start:window_start+(dt*2)] for ind, window_start in enumerate(list(range(0,x_full.shape[1],dt*2)))}

    print(f'{x_dict}')

    cov_dict = {}
    for x_title, x_all_time in x_dict.items():
        # Distance
        


        # pandas
        # # Build pandas
        # pandas_dict = {}
        # for i in range(x_all_time.shape[1]):
        #     pandas_dict[f'nirs_{i}'] = x_all_time[:,i]
        # for i in range(y_full.shape[1]):
        #     pandas_dict[f'eeg_{i}'] = y_full[:,i]
        # df = pd.DataFrame(pandas_dict)
        # print(df)
        # # Get covariance
        # c = df.cov()

        # Manual
        # c = np.zeros((x_all_time.shape[1],x_all_time.shape[1]))
        # for i in range(x_full.shape[0]):
        #     y = y_full[i,:]
        #     x = x_all_time[i,:]
        #     # Get covariance
        #     c_t = np.outer(x.T - x.mean(), y - y.mean())
        #     c = c + c_t/x_full.shape[0]
        #     # c = c + (c_t/x_full.shape[0])
        
        cov_dict[x_title] = c

    if plot:
        _plot_variance_corelation(wave_type, all_info, cov_dict, title='Covariance', subject_id=subject_id)

    return cov_dict, x_dict, all_info
'''