'''
This file contains the function to perform rolling correlation and plot the results
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.ndimage
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count

def process_time_bin(args):
    X, Y, gpMat_t, t = args
    Xgp = X * gpMat_t
    Ygp = Y * gpMat_t
    Xmovingavg = np.sum(X * gpMat_t, axis=1) / np.sum(gpMat_t)
    Ymovingavg = np.sum(Y * gpMat_t, axis=1) / np.sum(gpMat_t)
    
    Xgp = Xgp - np.outer(Xmovingavg, gpMat_t)
    Ygp = Ygp - np.outer(Ymovingavg, gpMat_t)
    
    covMat = np.dot(Xgp, Ygp.T)
    covMatX = np.dot(Xgp, Xgp.T)
    covMatY = np.dot(Ygp, Ygp.T)
    
    DX = np.sqrt(np.diag(1.0 / np.diag(covMatX)))
    DY = np.sqrt(np.diag(1.0 / np.diag(covMatY)))
    
    corMat = np.dot(np.dot(np.diag(DX), covMat), np.diag(DY))
    
    return corMat, covMat, covMatX, covMatY

def continuous_correlation2_tensor_looped(X, Y, L, numBins):
    N, T = X.shape
    M, _ = Y.shape
    
    gpMat = np.exp(-cdist(np.arange(1, T+1).reshape(-1, 1), np.linspace(1, T, numBins).reshape(-1, 1))**2 / (2 * L**2))
    
    results = [process_time_bin((X, Y, gpMat[:, t], t)) for t in range(numBins)]
    results_array = np.array(results)

    corMat = results_array[:, 0].transpose(1, 2, 0)
    covMat = results_array[:, 1].transpose(1, 2, 0)
    covMatX = results_array[:, 2].transpose(1, 2, 0)
    covMatY = results_array[:, 3].transpose(1, 2, 0)
    
    return corMat

# Function to perform rolling correlation and plot the results
def rolling_correlation(X, X_pred, chan_labels, offset, sampling_frequency, timeSigma, num_bins, zoom_start=None, zoom_end=None, do_legend=True, do_colorbar=True, ax=None, title=None):
    corMat = continuous_correlation2_tensor_looped(X, X_pred, timeSigma, num_bins)
    cor = np.zeros(X.shape)
    
    # Correcting zoom to handle one-dimensional array scaling
    for i in range(X.shape[0]):
        cor[i, :] = scipy.ndimage.zoom(corMat[i, i, :], X.shape[1] / corMat.shape[2], order=0)

    # plot single plot of X signal ontop X_pred signal colored by correlation
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    x_axis = np.arange(X.shape[1])
    y_axis_target = X + offset * np.arange(1, X.shape[0] + 1).reshape(-1, 1)
    y_axis_predicted = X_pred + offset * np.arange(1, X_pred.shape[0] + 1).reshape(-1, 1)
    # scatter = ax.scatter(x_axis.ravel(), y_axis_predicted.ravel(), c=cor.ravel(), cmap='viridis', s=1, vmin=-1, vmax=1)
    # scatter = ax.scatter(x_axis.ravel(), y_axis_target.ravel(), c=cor.ravel(), cmap='viridis', s=1, vmin=-1, vmax=1)
    # scatter = ax.scatter(x_axis.ravel(), y_axis_predicted.ravel(), s=1, vmin=-1, vmax=1, c='pink', alpha=0.8)

    for (data, y_axis, color) in zip([X, X_pred], [y_axis_target, y_axis_predicted], [ {'cmap':'viridis'}, {'color':'pink'},]):
        time = (np.arange(data.shape[1]) / sampling_frequency).reshape(1,-1)  # Convert index to time
        points = np.array([time, y_axis]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(-1, 1)
        if 'cmap' in color:
            lc = LineCollection(segments, cmap=color['cmap'], norm=norm, linewidth=2, label='Target')
            lc.set_array(cor.ravel())  # Color the lines by the correlation
        elif 'color' in color:
            # opacity = 0.5
            lc = LineCollection(segments, color=color['color'], norm=norm, linewidth=1, label='Predicted', alpha=0.5)

        ax.add_collection(lc, autolim=True)
    ax.set_yticks(offset * np.arange(X.shape[0]))
    ax.set_yticklabels(chan_labels)

    ax.set_ylim(min(y_axis_target.min(), y_axis_predicted.min()), max(y_axis_target.max(), y_axis_predicted.max()))
    if zoom_start is not None and zoom_end is not None:
        ax.set_xlim(zoom_start, zoom_end)
    else:
        ax.set_xlim(time.min(), time.max())

    ax.set_xlabel('time(s)')
    ax.set_ylabel('Channels')
    if do_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if do_colorbar:
        plt.colorbar(lc, ax=ax, label='Correlation')
    plt.tight_layout()

    average_correlation = np.mean(cor)

    return ax.get_figure(), average_correlation

def process_channel_rolling_correlation(channel_name, 
                                        i, 
                                        targets_train, 
                                        predictions_train, 
                                        targets_test, 
                                        predictions_test, 
                                        data_config):
    results = []
    for targets, predictions, type_label in [
        (targets_train, predictions_train, 'Train'),
        (targets_test, predictions_test, 'Test')
    ]:
        targets_single = targets[:,:,i].reshape(1, -1)
        predictions_single = predictions[:,:,i].reshape(1, -1)

        fig, average_correlation = rolling_correlation(
            targets_single, 
            predictions_single, 
            [channel_name], 
            offset=0,
            sampling_frequency=data_config['target_sample_rate'],
            timeSigma=100, 
            num_bins=50, 
            zoom_start=0, 
            do_legend=False,
            do_colorbar=False,
            ax=None, 
            title=''
        )
        results.append((type_label, average_correlation, fig))
    
    return channel_name, i, results
