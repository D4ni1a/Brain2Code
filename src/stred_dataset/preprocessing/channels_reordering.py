import numpy as np
from typing import List, Dict, Optional

def make_channels_reordering(all_channels, selected_channels):
    """
    Reorder EEG channels

    Args:
        all_channels: List of all 32 channels in original order
        selected_channels: List of 19 channels in desired order

    Returns:
        List of indices corresponding to selected_channels in the all_channels list
    """
    # Create a mapping from channel name to its index
    channel_to_idx = {channel: idx for idx, channel in enumerate(all_channels)}

    # Get indices of selected channels in the desired order
    selected_indices = []
    for channel in selected_channels:
        if channel not in channel_to_idx:
            raise ValueError(f"Channel {channel} not found in the list of all channels")
        selected_indices.append(channel_to_idx[channel])
    return selected_indices


def reorder_eeg_channels(eeg_data: np.ndarray,
                        selected_channels_indices: List[int]) -> np.ndarray:
    """
    Reorders and selects only the required EEG channels.

    Args:
        eeg_data: EEG data of shape (32, time_samples)
        selected_channels_indices: List of indices for the 19 channels in desired order

    Returns:
        EEG data of shape (19, time_samples)
    """
    # Select and reorder channels
    return eeg_data[selected_channels_indices, :]

def get_basic_channels():
    channels_order_data = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5',
                           'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
                           'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4',
                           'P8', 'O1', 'Iz', 'O2']
    channels_order_cbramod = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3',
                              'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

    selected_order = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                      0, 1]
    return channels_order_data, channels_order_cbramod, selected_order
