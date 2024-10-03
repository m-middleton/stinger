'''
This file contains the functions to format the data for the model.
'''

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class EEGfNIRSData(Dataset):
    def __init__(self, nirs_data, eeg_data):
        self.nirs_data = nirs_data
        self.eeg_data = eeg_data

    def __len__(self):
        return len(self.nirs_data)

    def __getitem__(self, idx):
        nirs_data = self.nirs_data[idx]
        eeg_data = self.eeg_data[idx]
        return nirs_data, eeg_data

def single_window_extraction(eeg_center,
                              eeg_data,
                              nirs_data,
                              nirs_sampling_rate,
                              eeg_sampling_rate,
                              eeg_i_min,
                              eeg_i_max,
                              nirs_i_min,
                              nirs_i_max):
    eeg_start = eeg_center + eeg_i_min
    eeg_end = eeg_center + eeg_i_max
    
    # Calculate corresponding NIRS center
    nirs_center = int(eeg_center * (nirs_sampling_rate / eeg_sampling_rate))
    nirs_start = nirs_center + nirs_i_min
    nirs_end = nirs_center + nirs_i_max
    
    # Extract EEG window
    single_eeg_window = eeg_data[:, eeg_start:eeg_end]
    # Extract NIRS window
    single_nirs_window = nirs_data[:, nirs_start:nirs_end]
    return single_eeg_window, single_nirs_window, eeg_start

def grab_ordered_windows(nirs_data,
                        eeg_data,
                        nirs_sampling_rate,
                        eeg_sampling_rate,
                        nirs_t_min, 
                        nirs_t_max,
                        eeg_t_min, 
                        eeg_t_max,
                        markers=None):
    # Calculate indices
    nirs_i_min = int(nirs_t_min * nirs_sampling_rate)
    nirs_i_max = int(nirs_t_max * nirs_sampling_rate)
    eeg_i_min = int(eeg_t_min * eeg_sampling_rate)
    eeg_i_max = int(eeg_t_max * eeg_sampling_rate)

    eeg_window_size = eeg_i_max - eeg_i_min
    nirs_window_size = nirs_i_max - nirs_i_min

    # Calculate the valid range for EEG window centers
    nirs_eeg_i_min = int(nirs_t_min * eeg_sampling_rate)
    nirs_eeg_i_max = int(nirs_t_max * eeg_sampling_rate)
    eeg_min_center = max(max(abs(eeg_i_min), 0), abs(nirs_eeg_i_min))
    eeg_max_center = eeg_data.shape[1] - max(max(eeg_i_max, 0), nirs_eeg_i_max)

    eeg_full_windows = []
    nirs_full_windows = []
    meta_data = []
    first_window_start = None

    for eeg_center in range(eeg_min_center, eeg_max_center, eeg_window_size):

        single_eeg_window, single_nirs_window, eeg_start = single_window_extraction(eeg_center=eeg_center,
                                                                          eeg_data=eeg_data,
                                                                          nirs_data=nirs_data,
                                                                          nirs_sampling_rate=nirs_sampling_rate,
                                                                          eeg_sampling_rate=eeg_sampling_rate,
                                                                          nirs_i_min=nirs_i_min,
                                                                          nirs_i_max=nirs_i_max,
                                                                          eeg_i_min=eeg_i_min,
                                                                          eeg_i_max=eeg_i_max,)
        
        # Check if NIRS window is within bounds
        if first_window_start is None:
            first_window_start = eeg_start
        
        meta_data.append(eeg_center)
        
        eeg_full_windows.append(single_eeg_window)
        nirs_full_windows.append(single_nirs_window)

    eeg_full_windows = np.array(eeg_full_windows)
    nirs_full_windows = np.array(nirs_full_windows)

    # Adjust marker timestamps
    if markers is not None and first_window_start is not None:
        markers[:, 0] -= first_window_start

    return eeg_full_windows, nirs_full_windows, meta_data, markers

def grab_random_windows(nirs_data,
                        eeg_data,
                        nirs_sampling_rate,
                        eeg_sampling_rate,
                        nirs_t_min, 
                        nirs_t_max,
                        eeg_t_min, 
                        eeg_t_max,
                        number_of_windows=1000):
    # Calculate indices
    nirs_i_min = int(nirs_t_min * nirs_sampling_rate)
    nirs_i_max = int(nirs_t_max * nirs_sampling_rate)
    eeg_i_min = int(eeg_t_min * eeg_sampling_rate)
    eeg_i_max = int(eeg_t_max * eeg_sampling_rate)

    # Calculate the valid range for EEG window centers
    nirs_eeg_i_min = int(nirs_t_min * eeg_sampling_rate)
    nirs_eeg_i_max = int(nirs_t_max * eeg_sampling_rate)
    eeg_min_center = max(max(abs(eeg_i_min), 0), abs(nirs_eeg_i_min))
    eeg_max_center = eeg_data.shape[1] - max(max(eeg_i_max, 0), nirs_eeg_i_max)

    print(eeg_min_center, eeg_max_center)
    print(nirs_eeg_i_min, nirs_eeg_i_max, eeg_i_min, eeg_i_max)

    eeg_full_windows = []
    nirs_full_windows = []
    meta_data = []

    for _ in range(number_of_windows):
        eeg_center = np.random.randint(eeg_min_center, eeg_max_center)
        single_eeg_window, single_nirs_window, eeg_start = single_window_extraction(eeg_center=eeg_center,
                                                                          eeg_data=eeg_data,
                                                                          nirs_data=nirs_data,
                                                                          nirs_sampling_rate=nirs_sampling_rate,
                                                                          eeg_sampling_rate=eeg_sampling_rate,
                                                                          nirs_i_min=nirs_i_min,
                                                                          nirs_i_max=nirs_i_max,
                                                                          eeg_i_min=eeg_i_min,
                                                                          eeg_i_max=eeg_i_max,)
        
        # Check if NIRS window is within bounds
        meta_data.append(eeg_center)
        
        eeg_full_windows.append(single_eeg_window)
        nirs_full_windows.append(single_nirs_window)

    eeg_full_windows = np.array(eeg_full_windows)
    nirs_full_windows = np.array(nirs_full_windows)

    return eeg_full_windows, nirs_full_windows, meta_data

def plot_series(target, output, epoch):
    plt.figure(figsize=(10, 4))
    plt.plot(target, label='Target')
    plt.plot(output, label='Output', linestyle='--')
    plt.title(f'Epoch {epoch + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_indices(x, y):
    indices = []
    for item in y:
        if item in x:
            indices.append(x.index(item))
    return indices