# Import necessary libraries
import os
from wav.wav_signal_processing import *
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import seaborn as sns
from config import *
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
#plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

def sort_breath_cycles_by_duration(full_breath_cycles, sample_rate):
    durations = [(end - start) / sample_rate for start, end in full_breath_cycles]
    sorted_indices = np.argsort(durations)
    return np.array(full_breath_cycles)[sorted_indices]

def filter_breaths_by_laser_events(full_breath_cycles, laser_events):
    """
    Filter the breath cycles to only include those that overlap with a laser event.
    
    Parameters:
        full_breath_cycles (numpy array): Array of tuples indicating the start and end of each breath cycle.
        laser_events (list): List of tuples indicating the start and end of each laser event.
        
    Returns:
        numpy array: Filtered array of breath cycles that overlap with laser events.
    """
    filtered_cycles = []
    for cycle in full_breath_cycles:
        for event in laser_events:
            if cycle[1] >= event[0] and cycle[0] <= event[1]:  # Check for overlap
                filtered_cycles.append(cycle)
                break  # No need to check other events for this cycle
    return np.array(filtered_cycles)

def plot_multiple_breath_phases(full_breath_cycles, expiration_indices, filtered_breathing_signal, laser_events, percent, plot_laser_event_cycles=False):
    """
    Plot breath cycles, highlighting those that are among the longest based on the given percentage.
    
    Parameters:
        full_breath_cycles (numpy array): Array of tuples indicating the start and end of each breath cycle.
        expiration_indices (numpy array): Array of expiration indices.
        filtered_breathing_signal (numpy array): The filtered breathing signal data.
        laser_events (list): List of tuples indicating the start and end of each laser event.
        percent (float): The percentage of the longest cycles to plot.
        plot_laser_event_cycles (bool): Whether to plot only cycles that overlap with a laser event.
    """
    
    # Filter by laser events if required
    if plot_laser_event_cycles:
        full_breath_cycles = filter_breaths_by_laser_events(full_breath_cycles, laser_events)
    
    # Sort breath cycles by duration
    sorted_cycles = sort_breath_cycles_by_duration(full_breath_cycles, sample_rate)
    
    # Determine the number of cycles to plot based on the provided percentage
    n_longest_percent = int((percent / 100) * len(sorted_cycles))
    
    # Select the longest cycles based on the determined number
    longest_percent_cycles = sorted_cycles[-n_longest_percent:]
    
    plt.figure(figsize=(12, 6))
    
    for cycle in longest_percent_cycles:
        cycle_start, cycle_end = cycle
        cycle_data = filtered_breathing_signal[cycle_start:cycle_end]
        plt.plot(cycle_data, alpha=0.5, label=f"Breath Cycle from {cycle_start} to {cycle_end}")
    
    # Set plot title based on whether laser event cycles are plotted
    title_suffix = "during Laser Events" if plot_laser_event_cycles else "All"
    plt.title(f"Longest {percent}% of Breath Cycles ({title_suffix})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

# Read the WAV file
sample_rate, audio_data = wavfile.read(FILE_PATH)
    
# Assign Channels
breathing_channel = audio_data[:, 0]
laser_channel = audio_data[:, 1]  

# Filter the breathing channel
filtered_breathing_signal = bandpass_filter(breathing_channel, 2, 50, sample_rate)
    
# Find inspirations in the filtered breathing signal
inspiration_indices = find_inspirations(filtered_breathing_signal, min_duration=0.01, min_amplitude=20)
    
# Find expirations and full breath cycles
expiration_indices = find_expirations(filtered_breathing_signal, inspiration_indices)
full_breath_cycles = find_full_breaths(inspiration_indices, expiration_indices)

# Find laser events for plotting
filtered_laser_events = find_laser_events(laser_channel, 1)

# Plot breath phases for two random example laser events
n_random_events = 2 # Number of random laser events you want to plot
random_laser_event_starts = random.sample([event[0] for event in filtered_laser_events], n_random_events)
for i, laser_event_start in enumerate(random_laser_event_starts, start=1):
    plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=f"Random #{i}")

    # Initialize lists to store laser_breath_phase and time_to_next_inspiration for each laser event
    laser_event_phases = []
    time_to_next_inspirations = []

    for event_start, event_end in filtered_laser_events:
        # Find the full breath cycle that contains the laser event
        containing_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]][0]
        
        # Find the full breath cycle that precedes the one containing the laser event
        preceding_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[1] < containing_breath_cycle[0]][-1]
        
        # Find the expiration index that corresponds to the inspiration start of the preceding breath cycle
        corresponding_expiration_start = expiration_indices[expiration_indices > preceding_breath_cycle[0]][0]
        
        # Calculate the accurate breath phases for the preceding breath cycle
        laser_event_phase = calculate_breath_phases(preceding_breath_cycle[0], corresponding_expiration_start, preceding_breath_cycle[1])
        
        # Calculate time to next inspiration from the start of the containing breath cycle
        time_to_next_insp, laser_breath_insp_start, next_insp_idx = time_to_next_inspiration_from_insp_start(event_start, inspiration_indices, sample_rate)
        
        laser_event_phases.append(laser_event_phase)
        time_to_next_inspirations.append(time_to_next_insp)
        
# Plot the longest 10% of breath cycles
plot_multiple_breath_phases(full_breath_cycles, expiration_indices, filtered_breathing_signal, filtered_laser_events, percent=1, plot_laser_event_cycles=False)

plot_multiple_breath_phases(full_breath_cycles, expiration_indices, filtered_breathing_signal, filtered_laser_events, percent=5, plot_laser_event_cycles=True)
