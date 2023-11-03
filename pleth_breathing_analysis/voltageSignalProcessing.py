__all__ = ['bandpass_filter', 'find_inspirations', 'find_expirations',
           'find_full_breaths','find_laser_events', 'calculate_breath_phases',
           'extend_and_match_breath_phase', 'find_laser_event_phases', 'calculate_breath_durations']

#import stuff
import pandas as pd 
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from config import *
import random

def bandpass_filter(signal, low_freq, high_freq, sample_rate):
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def find_inspirations(signal, sample_rate=1000, min_duration=0.02, max_duration=0.50, above_zero=None, min_amplitude=20):
    """
    Identify inspirations in the given signal with duration and amplitude thresholds.
    
    Parameters:
    - signal (numpy array): The input signal.
    - sample_rate (int): The sample rate of the signal.
    - min_duration (float): The minimum duration of an inspiration in seconds. Default is 0.03.
    - max_duration (float): The maximum duration of an inspiration in seconds. Default is 0.25.
    - min_amplitude (float): The minimum amplitude to consider an inspiration. Default is 0.1.
    - above_zero (list): Optional. A list specifying where the signal is above zero. If None, it is calculated.
    Returns:
    - numpy array: Indices where inspirations occur.
    """
    if above_zero is None:
        above_zero = signal > 0
        
    inspiration_indices = []
    possible_inspiration_start = None
    min_amplitude = min_amplitude

    for i in range(1, len(signal)):
        if above_zero[i] != above_zero[i - 1]:
            if not above_zero[i]:
                # A possible inspiration starts
                possible_inspiration_start = i
            elif possible_inspiration_start is not None:
                # A possible inspiration ends, check duration and amplitude
                duration = (i - possible_inspiration_start) / sample_rate
                amplitude = np.max(signal[possible_inspiration_start:i]) - np.min(signal[possible_inspiration_start:i])
                
                if min_duration <= duration <= max_duration and amplitude >= min_amplitude:
                    inspiration_indices.append(possible_inspiration_start)
                
                # Reset the possible inspiration start
                possible_inspiration_start = None
                
    return np.array(inspiration_indices)


# Find expirations function
def find_expirations(signal, inspiration_indices, above_zero=None):
    if above_zero is None:
        above_zero = []
    expiration_indices = []
    above_zero = signal > 0
    for insp_idx in inspiration_indices:
        for i in range(insp_idx + 1, len(signal)):
            if above_zero[i] != above_zero[i - 1] and above_zero[i]:
                expiration_indices.append(i)
                break
    return np.array(expiration_indices)

def find_full_breaths(inspiration_indices, expiration_indices):
    """
    Identify the start and end indices of full breath cycles based on inspirations and expirations.
    
    Parameters:
    - inspiration_indices (numpy array): Indices where inspirations occur.
    - expiration_indices (numpy array): Indices where expirations occur.
    
    Returns:
    - list of tuples: Each tuple contains the start and end index of a full breath cycle.
    """
    full_breath_cycles = []
    for insp_idx in inspiration_indices[:-1]:  # Exclude the last inspiration index
        # Find the closest expiration index that follows the current inspiration index
        exp_idx = expiration_indices[expiration_indices > insp_idx][0]
        
        # Find the next inspiration index that follows the current expiration index
        next_insp_idx = inspiration_indices[inspiration_indices > exp_idx][0]
        
        # Store the start and end indices of the full breath cycle
        full_breath_cycles.append((insp_idx, next_insp_idx))
        
    return full_breath_cycles

# Identify the laser events you want to analyze.
def find_laser_events(laser_channel, event_duration_ms, laser_events=None):
    if laser_events is None:
        laser_events = []
    # Set a robust threshold for the laser channel
    max_laser_value = np.max(laser_channel)
    robust_laser_threshold = max_laser_value * 0.01
    event_duration_ms = EVENT_DURATION_MS
    sample_rate = SAMPLE_RATE
    # Identify laser events
    laser_events = []
    laser_event_start = None
    for i in range(1, len(laser_channel)):
        if laser_channel[i] > robust_laser_threshold:
            if laser_event_start is None:
                laser_event_start = i
        else:
            if laser_event_start is not None:
                laser_events.append((laser_event_start, i))
                laser_event_start = None
    # Filter laser events based on duration
    filtered_laser_events = []
    filtered_laser_events = [(start, end) for start, end in laser_events if np.isclose((end - start) / sample_rate, event_duration_ms / 1000, atol=0.001)]
    return filtered_laser_events

# Function to calculate breath phases for a full breath cycle accurately
def calculate_breath_phases(insp_start, exp_start, insp_end):
    insp_length = exp_start - insp_start
    exp_length = insp_end - exp_start
    
    insp_phase = np.linspace(0, np.pi, insp_length)
    exp_phase = np.linspace(np.pi, 2 * np.pi, exp_length)
    
    laser_breath_phase = []
    laser_breath_phase = np.concatenate((insp_phase, exp_phase))
    return laser_breath_phase

def extend_and_match_breath_phase(breath_phases, start_idx, end_idx, laser_event_time):
    """
    Extend the breath phase array to match the duration until the laser event time.
    Each time the pattern repeats, add 2pi to maintain phase continuity.

    Parameters:
    - breath_phases: The breath phases for a single breath cycle.
    - start_idx: The start index of the breath cycle.
    - end_idx: The end index of the breath cycle.
    - laser_event_time: The time/index of the laser event.

    Returns:
    - Extended breath phases array.
    - The breath phase at the time of the laser event.
    """
    cycle_length = len(breath_phases)
    extended_length = laser_event_time - start_idx
    num_repeats = extended_length // cycle_length

    # Extend the breath phases
    extended_phases = np.tile(breath_phases, num_repeats + 1)

    # Add 2pi for each repeat
    for i in range(1, num_repeats + 1):
        extended_phases[i * cycle_length: (i + 1) * cycle_length] += 2 * np.pi * i

    # Cut the array to match the exact length needed
    extended_phases = extended_phases[:extended_length]

    # Get the phase at the time of the laser event
    laser_event_phase = extended_phases[-1]

    return extended_phases, laser_event_phase

# Function to process all laser events and store their phases
def find_laser_event_phases(laser_events, full_breath_cycles, expiration_indices, filtered_breathing_signal):
    laser_event_phases = []
    for laser_event_start in [event[0] for event in laser_events]:
        preceding_full_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[1] <= laser_event_start][-1]
        preceding_expiration_start = expiration_indices[expiration_indices > preceding_full_breath_cycle[0]][0]
        preceding_accurate_breath_phases = calculate_breath_phases(preceding_full_breath_cycle[0], preceding_expiration_start, preceding_full_breath_cycle[1])

        # Extend and match the breath phase
        _, laser_event_phase = extend_and_match_breath_phase(preceding_accurate_breath_phases, preceding_full_breath_cycle[0], preceding_full_breath_cycle[1], laser_event_start)
        
        # Subtract 2pi radians from the phase before storing
        adjusted_phase = laser_event_phase - 2 * np.pi

        # Store the adjusted phase
        laser_event_phases.append(adjusted_phase)
    
    return laser_event_phases

def calculate_breath_durations(containing_breath_cycles_for_laser_events, full_breath_cycles, sample_rate, num_preceding_breaths=1):
    """
    Calculate the duration of breaths that precede laser breaths and the duration of laser breaths.
    
    Parameters:
    - containing_breath_cycles_for_laser_events: List of tuples containing start and end indices of breaths containing laser events.
    - full_breath_cycles: List of tuples containing start and end indices of all breaths.
    - sample_rate: Sampling rate of the data (in Hz).
    - num_preceding_breaths: Number of breaths leading up to the laser breath to average for the preceding breath duration.
    
    Returns:
    - avg_preceding_breath_durations: List of averaged durations (in seconds) of breaths that precede laser breaths.
    - laser_breath_durations: List of durations (in seconds) of laser breaths.
    """
    
    # Initialize lists to store durations
    avg_preceding_breath_durations = []
    laser_breath_durations = []
    
    # Iterate through breath cycles containing laser events
    for laser_breath in containing_breath_cycles_for_laser_events:
        # Calculate the duration of the laser breath
        laser_breath_duration = (laser_breath[1] - laser_breath[0]) / sample_rate
        laser_breath_durations.append(laser_breath_duration)
        
        # Find the preceding breath cycles
        preceding_breaths = []
        count = 0
        for full_breath in reversed(full_breath_cycles):
            if full_breath[1] < laser_breath[0]:
                preceding_breaths.append(full_breath)
                count += 1
                if count >= num_preceding_breaths:
                    break
        
        # Calculate the average duration of the preceding breath cycles
        if preceding_breaths:
            avg_preceding_duration = np.mean([(b[1] - b[0]) / sample_rate for b in preceding_breaths])
            avg_preceding_breath_durations.append(avg_preceding_duration)
    
    return avg_preceding_breath_durations, laser_breath_durations