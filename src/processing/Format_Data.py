'''
This file contains the functions to format the data for the model.
'''

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SignalData(Dataset):
    """
    A PyTorch Dataset class for handling input and target signals.
    """
    def __init__(self, input_signal_data, target_signal_data):
        self.input_signal_data = input_signal_data
        self.target_signal_data = target_signal_data

    def __len__(self):
        return len(self.input_signal_data)

    def __getitem__(self, idx):
        input_data = self.input_signal_data[idx]
        target_data = self.target_signal_data[idx]
        return input_data, target_data

def single_window_extraction(center_index,
                             input_signal,
                             target_signal,
                             input_sample_rate,
                             target_sample_rate,
                             input_i_min,
                             input_i_max,
                             target_i_min,
                             target_i_max):
    """
    Extracts a single window from the input and target signals based on the center index.

    Parameters:
    - center_index: The center index in the input signal
    - input_signal: numpy array of input signal data (channels x samples)
    - target_signal: numpy array of target signal data (channels x samples)
    - input_sampling_rate: Sampling rate of the input signal
    - target_sampling_rate: Sampling rate of the target signal
    - input_i_min: Minimum index offset for input signal window
    - input_i_max: Maximum index offset for input signal window
    - target_i_min: Minimum index offset for target signal window
    - target_i_max: Maximum index offset for target signal window

    Returns:
    - input_window: Extracted window from the input signal
    - target_window: Extracted window from the target signal
    - input_start: Start index of the input signal window
    """
    target_start = center_index + target_i_min
    target_end = center_index + target_i_max

    # Calculate corresponding target center index
    input_center = int(center_index * (input_sample_rate / target_sample_rate))
    input_start = input_center + input_i_min
    input_end = input_center + input_i_max

    # Extract windows
    input_window = input_signal[:, input_start:input_end]
    target_window = target_signal[:, target_start:target_end]

    return input_window, target_window, target_start

def grab_ordered_windows(input_signal,
                         target_signal,
                         input_sample_rate,
                         target_sample_rate,
                         input_t_min,
                         input_t_max,
                         target_t_min,
                         target_t_max,
                         events=None):
    """
    Extracts ordered windows from the input and target signals.

    Parameters:
    - input_signal: numpy array of input signal data (channels x samples)
    - target_signal: numpy array of target signal data (channels x samples)
    - input_sampling_rate: Sampling rate of the input signal
    - target_sampling_rate: Sampling rate of the target signal
    - input_t_min: Start time offset for input signal window
    - input_t_max: End time offset for input signal window
    - target_t_min: Start time offset for target signal window
    - target_t_max: End time offset for target signal window
    - events: Optional array of events to adjust timestamps

    Returns:
    - input_full_windows: Array of input signal windows
    - target_full_windows: Array of target signal windows
    - meta_data: List of center indices used for window extraction
    - events: Adjusted events array (if provided)
    """
    # Calculate indices for input signal
    input_i_min = int(input_t_min * input_sample_rate)
    input_i_max = int(input_t_max * input_sample_rate)
    input_window_size = input_i_max - input_i_min

    # Calculate indices for target signal
    target_i_min = int(target_t_min * target_sample_rate)
    target_i_max = int(target_t_max * target_sample_rate)
    target_window_size = target_i_max - target_i_min

    # Calculate the valid range for target signal window centers
    input_target_i_min = int(input_t_min * target_sample_rate)
    input_target_i_max = int(input_t_max * target_sample_rate)
    target_min_center = max(max(abs(target_i_min), 0), abs(input_target_i_min))
    target_max_center = target_signal.shape[1] - max(max(target_i_max, 0), input_target_i_max)

    input_full_windows = []
    target_full_windows = []
    meta_data = []
    first_window_start = None

    for center_index in range(target_min_center, target_max_center, target_window_size):
        input_window, target_window, target_start = single_window_extraction(
            center_index=center_index,
            input_signal=input_signal,
            target_signal=target_signal,
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_i_min=input_i_min,
            input_i_max=input_i_max,
            target_i_min=target_i_min,
            target_i_max=target_i_max
        )

        # Check if windows are within bounds
        if input_window.shape[1] != input_window_size or target_window.shape[1] != target_window_size:
            continue  # Skip incomplete windows at the edges

        if first_window_start is None:
            first_window_start = target_start

        meta_data.append(center_index)
        input_full_windows.append(input_window)
        target_full_windows.append(target_window)

    input_full_windows = np.array(input_full_windows)
    target_full_windows = np.array(target_full_windows)

    # Adjust event timestamps
    if events is not None and first_window_start is not None:
        events[:, 0] -= first_window_start

    return input_full_windows, target_full_windows, meta_data, events

def grab_random_windows(input_signal,
                        target_signal,
                        input_sample_rate,
                        target_sample_rate,
                        input_t_min,
                        input_t_max,
                        target_t_min,
                        target_t_max,
                        number_of_windows=1000):
    """
    Extracts random windows from the input and target signals.

    Parameters:
    - input_signal: numpy array of input signal data (channels x samples)
    - target_signal: numpy array of target signal data (channels x samples)
    - input_sampling_rate: Sampling rate of the input signal
    - target_sampling_rate: Sampling rate of the target signal
    - input_t_min: Start time offset for input signal window
    - input_t_max: End time offset for input signal window
    - target_t_min: Start time offset for target signal window
    - target_t_max: End time offset for target signal window
    - number_of_windows: Number of random windows to extract

    Returns:
    - input_full_windows: Array of input signal windows
    - target_full_windows: Array of target signal windows
    - meta_data: List of center indices used for window extraction
    """
    # Calculate indices for input signal
    input_i_min = int(input_t_min * input_sample_rate)
    input_i_max = int(input_t_max * input_sample_rate)
    input_window_size = input_i_max - input_i_min

    # Calculate indices for target signal
    target_i_min = int(target_t_min * target_sample_rate)
    target_i_max = int(target_t_max * target_sample_rate)
    target_window_size = target_i_max - target_i_min

    # Calculate the valid range for input signal window centers
    input_target_i_min = int(input_t_min * target_sample_rate)
    input_target_i_max = int(input_t_max * target_sample_rate)
    target_min_center = max(max(abs(target_i_min), 0), abs(input_target_i_min))
    target_max_center = target_signal.shape[1] - max(max(target_i_max, 0), input_target_i_max)

    print(f"Window center range: {target_min_center} to {target_max_center}")
    print(f"Input window indices: i_min={input_i_min}, i_max={input_i_max}")
    print(f"Target window indices: i_min={target_i_min}, i_max={target_i_max}")

    input_full_windows = []
    target_full_windows = []
    meta_data = []

    for _ in range(number_of_windows):
        center_index = np.random.randint(target_min_center, target_max_center)
        input_window, target_window, _  = single_window_extraction(
            center_index=center_index,
            input_signal=input_signal,
            target_signal=target_signal,
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            input_i_min=input_i_min,
            input_i_max=input_i_max,
            target_i_min=target_i_min,
            target_i_max=target_i_max
        )

        # Check if windows are within bounds
        if input_window.shape[1] != input_window_size or target_window.shape[1] != target_window_size:
            continue  # Skip incomplete windows at the edges

        meta_data.append(center_index)
        input_full_windows.append(input_window)
        target_full_windows.append(target_window)

    input_full_windows = np.array(input_full_windows)
    target_full_windows = np.array(target_full_windows)

    return input_full_windows, target_full_windows, meta_data

def plot_series(target, output, epoch):
    """
    Plots the target and output series for visualization.

    Parameters:
    - target: The target signal series
    - output: The output signal series (model predictions)
    - epoch: The current epoch number (for labeling)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(target, label='Target')
    plt.plot(output, label='Output', linestyle='--')
    plt.title(f'Epoch {epoch + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_indices(x, y):
    """
    Finds the indices of items in list y that are present in list x.

    Parameters:
    - x: The list to search within
    - y: The list of items to find

    Returns:
    - indices: List of indices corresponding to items in y found in x
    """
    indices = []
    for item in y:
        if item in x:
            indices.append(x.index(item))
    return indices
