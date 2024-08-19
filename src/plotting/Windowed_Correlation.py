'''
This file contains the function to perform rolling correlation and plot the results
'''

import scipy.ndimage
from scipy.spatial.distance import cdist

from matplotlib.collections import LineCollection

import numpy as np
import matplotlib.pyplot as plt

def continuous_correlation2_tensor_looped(X, Y, L, numBins, multipliermethod='multiprod'):
    N, T1 = X.shape
    M, T2 = Y.shape
    if T1 != T2:
        raise ValueError('columns have to match')
    else:
        T = T1
    
    # Gaussian process matrix with weights to nearby time points
    gpMat = np.exp(-cdist(np.arange(1, T+1).reshape(-1, 1), np.linspace(1, T, numBins).reshape(-1, 1))**2 / (2 * L**2))
    
    covMat = np.zeros((N, M, numBins))
    corMat = np.zeros((N, M, numBins))
    covMatX = np.zeros((N, N, numBins))
    covMatY = np.zeros((M, M, numBins))
    
    for t in range(numBins):
        Xgp = X * gpMat[:, t]
        Ygp = Y * gpMat[:, t]
        Xmovingavg = np.sum(X * gpMat[:, t], axis=1) / np.sum(gpMat[:, t])
        Ymovingavg = np.sum(Y * gpMat[:, t], axis=1) / np.sum(gpMat[:, t])
        
        Xgp = Xgp - np.outer(Xmovingavg, gpMat[:, t])
        Ygp = Ygp - np.outer(Ymovingavg, gpMat[:, t])
        
        covMat[:, :, t] = np.dot(Xgp, Ygp.T)
        covMatX[:, :, t] = np.dot(Xgp, Xgp.T)
        covMatY[:, :, t] = np.dot(Ygp, Ygp.T)
        
        DX = np.sqrt(np.diag(1.0 / np.diag(covMatX[:, :, t])))
        DY = np.sqrt(np.diag(1.0 / np.diag(covMatY[:, :, t])))
        
        corMat[:, :, t] = np.dot(np.dot(np.diag(DX), covMat[:, :, t]), np.diag(DY))
    
    return corMat

# Function to perform rolling correlation and plot the results
def rolling_correlation_two_sided(X, X_pred, chan_labels, offset, sampling_frequency, timeSigma, num_bins, zoom_start=None, zoom_end=None, ax=None):
    corMat = continuous_correlation2_tensor_looped(X, X_pred, timeSigma, num_bins)
    cor = np.zeros(X.shape)
    
    # Correcting zoom to handle one-dimensional array scaling
    for i in range(X.shape[0]):
        cor[i, :] = scipy.ndimage.zoom(corMat[i, i, :], X.shape[1] / corMat.shape[2], order=0)

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    x_axis = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))
    y_axis_target = X + offset * np.arange(1, X.shape[0] + 1).reshape(-1, 1)
    y_axis_predicted = X_pred + offset * np.arange(1, X_pred.shape[0] + 1).reshape(-1, 1)
    
    for idx, (data, y_axis) in enumerate(zip([X, X_pred], [y_axis_target, y_axis_predicted])):
        scatter = ax[idx].scatter(x_axis.ravel(), y_axis.ravel(), c=cor.ravel(), cmap='viridis', s=1, vmin=-1, vmax=1)
        ax[idx].set_title('Target' if idx == 0 else 'Predicted')
        ax[idx].set_xlabel('time(s)')
        ax[idx].set_ylabel('Channels')
        ax[idx].set_yticks(offset * np.arange(data.shape[0]))
        ax[idx].set_yticklabels(chan_labels)
        ax[idx].set_xticklabels(np.round(ax[idx].get_xticks() / sampling_frequency, 2))
        plt.colorbar(scatter, ax=ax[idx], label='Correlation')
    
    plt.tight_layout()
    plt.show()

    return fig

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
            lc = LineCollection(segments, color=color['color'], norm=norm, linewidth=1, label='Predicted', alpha=0.8)

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

    return ax.get_figure()
