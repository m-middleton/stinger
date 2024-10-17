import time
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

def calculate_channel_distances(target_coords, input_coords):
    target_positions = np.array(target_coords)
    input_positions = np.array(input_coords)
    return cdist(input_positions, target_positions)

def translate_channel_name_to_ch_id(channel_names, nirs_coords, channel_ids):
    translated_ids = []
    
    # Strip suffixes and get unique IDs and keep original order
    seen = set()
    unique_ids_list = [id[:-4] for id in channel_ids if id[:-4] not in seen and not seen.add(id[:-4])]
    lower_nirs_keys = [key.lower() for key in nirs_coords.keys()]
    assert len(unique_ids_list) == len(nirs_coords.keys())

    for channel_name in channel_names:
        channel_name_lower = channel_name.lower()
        if (channel_name_lower in lower_nirs_keys):
            coords_index = lower_nirs_keys.index(channel_name_lower)
            if unique_ids_list[coords_index] not in translated_ids:
                translated_ids.append(unique_ids_list[coords_index])
            else:
                print(f'Channel name {channel_name} already in translations')
        else:
            print(f'Channel name {channel_name} not found in NIRS_COORDS')
        
    return translated_ids

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