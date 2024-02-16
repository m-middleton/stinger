
def translate_channel_name_to_ch_id(nirs_coords, channel_names, channel_ids):
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