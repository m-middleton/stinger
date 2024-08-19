'''
This file contains the functions to format the data for the model.
'''

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class EEGfNIRSData(Dataset):
    def __init__(self, fnirs_data, eeg_data):
        self.fnirs_data = fnirs_data
        self.eeg_data = eeg_data
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.fnirs_data[idx], self.eeg_data[idx]

def get_single_window(center_point, 
                      nirs_data, 
                      eeg_data, 
                      eeg_i_min, 
                      eeg_i_max, 
                      nirs_i_min, 
                      nirs_i_max):
    eeg_low_index = center_point + eeg_i_min
    eeg_high_index = center_point + eeg_i_max
    single_eeg_window = eeg_data[:,eeg_low_index:eeg_high_index]

    nirs_low_index = center_point + nirs_i_min
    nirs_high_index = center_point + nirs_i_max
    single_nirs_window = nirs_data[:,nirs_low_index:nirs_high_index]
    
    return single_eeg_window, single_nirs_window

def grab_ordered_windows(nirs_data,
                        eeg_data,
                        sampling_rate,
                        nirs_t_min, 
                        nirs_t_max,
                        eeg_t_min, 
                        eeg_t_max):
    nirs_i_min = int(nirs_t_min*sampling_rate)
    nirs_i_max = int(nirs_t_max*sampling_rate)
    eeg_i_min = int(eeg_t_min*sampling_rate)
    eeg_i_max = int(eeg_t_max*sampling_rate)

    eeg_window_size = eeg_i_max - eeg_i_min

    max_center_eeg = eeg_data.shape[1] - eeg_i_max
    max_center_nirs = nirs_data.shape[1] - nirs_i_max
    max_center = np.min([max_center_eeg, max_center_nirs])

    min_center_eeg = np.abs(eeg_i_min)
    min_center_nirs = np.abs(nirs_i_min)
    min_center = np.max([min_center_eeg, min_center_nirs])

    nirs_full_windows = []
    eeg_full_windows = []
    meta_data = []

    for i in range(min_center, max_center, eeg_window_size):
        center_point = i
        meta_data.append(center_point)
        single_eeg_window, single_nirs_window = get_single_window(center_point, 
                                                                  nirs_data, 
                                                                  eeg_data, 
                                                                  eeg_i_min, 
                                                                  eeg_i_max, 
                                                                  nirs_i_min, 
                                                                  nirs_i_max)
        
        eeg_full_windows.append(single_eeg_window)
        nirs_full_windows.append(single_nirs_window)

    nirs_full_windows = np.array(nirs_full_windows)
    eeg_full_windows = np.array(eeg_full_windows)

    return eeg_full_windows, nirs_full_windows, meta_data
    
def grab_random_windows(nirs_data, 
                        eeg_data,
                        sampling_rate,
                        nirs_t_min, 
                        nirs_t_max,
                        eeg_t_min, 
                        eeg_t_max,
                        number_of_windows=1000):
    '''make number_of_windows of size t_min to t_max for each offset 0 to offset_t for eeg and nirs'''

    nirs_i_min = int(nirs_t_min*sampling_rate)
    nirs_i_max = int(nirs_t_max*sampling_rate)
    eeg_i_min = int(eeg_t_min*sampling_rate)
    eeg_i_max = int(eeg_t_max*sampling_rate)

    max_center_eeg = eeg_data.shape[1] - eeg_i_max
    max_center_nirs = nirs_data.shape[1] - nirs_i_max
    max_center = np.min([max_center_eeg, max_center_nirs])

    min_center_eeg = np.abs(eeg_i_min)
    min_center_nirs = np.abs(nirs_i_min)
    min_center = np.max([min_center_eeg, min_center_nirs])

    nirs_full_windows = []
    eeg_full_windows = []
    meta_data = []
    for i in range(number_of_windows):
        center_point = np.random.randint(min_center, max_center)
        meta_data.append(center_point)
        single_eeg_window, single_nirs_window = get_single_window(center_point, 
                                                                  nirs_data, 
                                                                  eeg_data, 
                                                                  eeg_i_min, 
                                                                  eeg_i_max, 
                                                                  nirs_i_min, 
                                                                  nirs_i_max)
        
        eeg_full_windows.append(single_eeg_window)
        nirs_full_windows.append(single_nirs_window)
    
    nirs_full_windows = np.array(nirs_full_windows)
    eeg_full_windows = np.array(eeg_full_windows)

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