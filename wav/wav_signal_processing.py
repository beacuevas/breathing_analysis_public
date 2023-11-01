__all__ = ['bandpass_filter', 'find_inspirations', 'find_expirations',
           'find_full_breaths', 'time_to_next_inspiration_from_insp_start',
           'avg_time_to_next_inspiration_N_breaths_before', 'assign_breath_phase', 
           'find_laser_events','calculate_breath_phases','plot_breath_phases', 
           'statistical_direct_validation', 'temporal_consistency_check',
           'calculate_breath_durations','plot_cycle_lengths_vs_phase', 
           'create_box_and_whisker_plot', 'plot_laser_and_breath_events']

#import stuff
import os

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
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

sample_rate = SAMPLE_RATE

# Bandpass filter function
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
    signal: Unit-less or specific to your application (e.g., voltage, pressure, etc.)
    sample_rate: Samples per second (Hz)
    min_duration: Seconds
    max_duration: Seconds
    min_amplitude: Same as the unit of the signal
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

# Time to next inspiration function
def time_to_next_inspiration_from_insp_start(event_start, inspiration_indices, sample_rate):
    time_to_next_insp = []
    laser_breath_insp_start = []
    next_insp_idx = []
    laser_breath_insp_start = inspiration_indices[inspiration_indices <= event_start][-1]
    next_insp_idx = inspiration_indices[inspiration_indices > event_start][0]
    time_to_next_insp = (next_insp_idx - laser_breath_insp_start) / sample_rate
    return time_to_next_insp, laser_breath_insp_start, next_insp_idx

# Average time to next inspiration function
def avg_time_to_next_inspiration_N_breaths_before(sample, inspiration_indices, sample_rate, num_breaths_before=5):
    avg_time = []
    prev_insp_indices = []
    times_to_next_insp = []
    prev_insp_indices = inspiration_indices[inspiration_indices < sample][-num_breaths_before:]
    times_to_next_insp = np.diff(prev_insp_indices) / sample_rate
    avg_time = np.mean(times_to_next_insp)
    return avg_time

# Optimized Assign corrected breath phase function
def assign_breath_phase(inspiration_indices, signal_length):
    breath_phases = np.zeros(signal_length, dtype=float)
    for i in range(len(inspiration_indices) - 1):
        insp_start = inspiration_indices[i]
        insp_end = inspiration_indices[i + 1]
        cycle_length = insp_end - insp_start

        if cycle_length <= 0:
            continue

        insp_phase = np.linspace(0, np.pi, cycle_length // 2)
        exp_phase = np.linspace(np.pi, 2 * np.pi, cycle_length - len(insp_phase))
        complete_breath_phase = np.concatenate((insp_phase, exp_phase))
        breath_phases[insp_start:insp_end] = complete_breath_phase[:len(complete_breath_phase)]
    return breath_phases

# Identify the laser events you want to analyze.
def find_laser_events(laser_channel, event_duration_ms, laser_events=None):
    if laser_events is None:
        laser_events = []
    # Set a robust threshold for the laser channel
    max_laser_value = np.max(laser_channel)
    robust_laser_threshold = max_laser_value * 0.01
    event_duration_ms = EVENT_DURATION_MS

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

# Function to plot accurate breath phases for given laser events using previous function outputs
def plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=''):
    # Find the full breath cycle that contains the laser event
    example_full_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[0] <= laser_event_start <= cycle[1]][0]
    
    # Find the expiration index that corresponds to the inspiration start of the example full breath cycle
    example_expiration_start = expiration_indices[expiration_indices > example_full_breath_cycle[0]][0]
    
    # Calculate the accurate breath phases for the example full breath cycle
    example_accurate_breath_phases = calculate_breath_phases(example_full_breath_cycle[0], example_expiration_start, example_full_breath_cycle[1])
    
    # Plot the accurate breath phases for the example laser event
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_breathing_signal[example_full_breath_cycle[0]:example_full_breath_cycle[1]], label="Filtered Breathing Signal")
    plt.axvline(x=laser_event_start - example_full_breath_cycle[0], color='m', linestyle='--', label=f"Laser Event Start")
    plt.twinx()
    plt.plot(example_accurate_breath_phases, 'g--', label="Breath Phases (in radians)")
    plt.ylabel("Phase (radians)")
    plt.legend(loc="upper right")
    plt.title(f"Breath Phase for Example Laser Event {title_suffix}")
    plt.show()

# Statistical direct validation
def statistical_direct_validation(laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events, sample_size=50):
    """
    Perform statistical direct validation to check if each set of elements across the lists corresponds to the same laser event.
    
    Parameters:
    - laser_breath_phases (list): List of numpy arrays containing the breath phase for each breath cycle containing a laser event.
    - filtered_laser_events (list): List of tuples containing the start and end indices of each laser event.
    - containing_breath_cycles_for_laser_events (list): List of tuples containing the start and end indices of the breath cycle for each laser event.
    - sample_size (int): Number of random samples to validate. Default is 50.
    
    Returns:
    - bool: True if all checks passed, False otherwise.
    """
    
    total_events = len(filtered_laser_events)
    
    # Randomly sample indices for validation
    sample_indices = random.sample(range(total_events), min(sample_size, total_events))
    
    for i in sample_indices:
        event_start, _ = filtered_laser_events[i]
        cycle_start, cycle_end = containing_breath_cycles_for_laser_events[i]
        laser_breath_phase = laser_breath_phases[i]
        
        # Check if event_start falls within the corresponding cycle
        if not (cycle_start <= event_start <= cycle_end):
            return False, f"Mismatch at index {i}: Event start {event_start} not in cycle {cycle_start}-{cycle_end}."
        
        # Check if the length of the laser_breath_phase array matches the cycle length
        if len(laser_breath_phase) != (cycle_end - cycle_start):
            return False, f"Mismatch at index {i}: Length of laser_breath_phase ({len(laser_breath_phase)}) not equal to cycle length ({cycle_end - cycle_start})."
    
    return True, "All checks passed."

# Temporal consistency check
def temporal_consistency_check(filtered_laser_events, containing_breath_cycles_for_laser_events):
    """
    Check if the laser events and breath cycles are in ascending order.
    
    Parameters:
    - filtered_laser_events (list): List of tuples containing the start and end indices of each laser event.
    - containing_breath_cycles_for_laser_events (list): List of tuples containing the start and end indices of the breath cycle for each laser event.
    
    Returns:
    - bool: True if all checks passed, False otherwise.
    - str: Message indicating the result.
    """
    
    last_event_start = -1
    last_cycle_start = -1

    for (event_start, _), (cycle_start, _) in zip(filtered_laser_events, containing_breath_cycles_for_laser_events):
        if event_start <= last_event_start or cycle_start <= last_cycle_start:
            return False, "Events or cycles are not in ascending order."
        last_event_start = event_start
        last_cycle_start = cycle_start

    return True, "All checks passed."

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


def plot_cycle_lengths_vs_phase(preceding_lengths, laser_lengths, phases):
    """
    Plot preceding_breath_cycle_length and laser_breath_cycle_length against laser_event_phase_values.
    
    Parameters:
    - preceding_lengths (list or numpy array): Lengths of preceding breath cycles.
    - laser_lengths (list or numpy array): Lengths of breath cycles containing laser events.
    - phases (list or numpy array): Laser event phase values.
    """
    
    # Check if all arrays have the same length
    if len(preceding_lengths) != len(laser_lengths) or len(laser_lengths) != len(phases):
        raise ValueError("All arrays must have the same length.")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot preceding_breath_cycle_length
    sns.scatterplot(x=phases, y=preceding_lengths, label='Preceding Breath Cycle Length', marker='o')
    
    # Plot laser_breath_cycle_length
    sns.scatterplot(x=phases, y=laser_lengths, label='Laser Breath Cycle Length', marker='x')
    
    # Add labels and title
    plt.xlabel('Laser Event Phase (in radians)')
    plt.ylabel('Breath Cycle Length (in seconds)')
    plt.title('Breath Cycle Length vs. Laser Event Phase')
    plt.legend()
    
    # Show the plot
    plt.show()

def create_box_and_whisker_plot(preceding_breath_durations=None, laser_breath_durations=None, laser_event_phase_values=None, num_bins=10, exclude_data=False):
    """
    Create a box and whisker plot based on the given data.

    Parameters:
    - preceding_breath_durations: List of preceding breath durations.
    - laser_breath_durations: List of laser breath durations.
    - laser_event_phase_values: List of laser event phase values.
    - num_bins: Number of bins to divide the laser_event_phase_values.
    - exclude_data: Boolean indicating whether to exclude data points for equal bin sizes.

    Returns:
    - A Matplotlib box and whisker plot with legend.
    """
    if preceding_breath_durations is None:
        preceding_breath_durations = []
    if laser_breath_durations is None:
        laser_breath_durations = []
    if laser_event_phase_values is None:
        laser_event_phase_values = []

    # Check for equal length of all lists
    if len(preceding_breath_durations) != len(laser_breath_durations) or len(laser_breath_durations) != len(laser_event_phase_values):
        raise ValueError("All lists must have the same length.")

    # Step 1: Group data into bins
    min_phase = 0
    max_phase = 2 * np.pi  # Adjusted to 2pi
    bin_edges = np.linspace(min_phase, max_phase, num_bins + 1)
    bin_indices = np.digitize(laser_event_phase_values, bin_edges)
    
    bins = {}
    for i, bin_idx in enumerate(bin_indices):
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append((preceding_breath_durations[i], laser_breath_durations[i]))

    # Step 2: Optionally make sure each bin has an equal number of data points
    if exclude_data:
        min_bin_size = min(len(b) for b in bins.values())
        total_removed_points = 0  # Initialize the count of removed points

        for bin_idx in bins.keys():
            num_removed_points = len(bins[bin_idx]) - min_bin_size  # Calculate the number of points to be removed from this bin
            total_removed_points += num_removed_points  # Update the total count
            random.shuffle(bins[bin_idx])  # Shuffle the tuples as whole units
            bins[bin_idx] = bins[bin_idx][:min_bin_size]  # Truncate the list of tuples

        print(f"Total number of data points removed: {total_removed_points}")  # Report the total number of removed points

    # Step 3: Create the box and whisker plot
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
    preceding_data = [[preceding for preceding, _ in bins.get(i+1, [])] for i in range(num_bins)]
    laser_data = [[laser for _, laser in bins.get(i+1, [])] for i in range(num_bins)]
    bin_labels = [f"{bin_centers[i]:.2f}" for i in range(num_bins)]

    fig, ax = plt.subplots()
    # Get the first two colors from the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_preceding = default_colors[0]
    color_laser = default_colors[1]

    # Create boxplots with patch_artist=True
    bp1 = ax.boxplot(preceding_data, positions=bin_centers, widths=np.diff(bin_edges), patch_artist=True, labels=bin_labels, showfliers=False)
    bp2 = ax.boxplot(laser_data, positions=bin_centers, widths=np.diff(bin_edges), patch_artist=True, labels=bin_labels, showfliers=False)

    # Set a consistent color for each group across all bins
    for box in bp1['boxes']:
        box.set_facecolor(color_preceding)
    for box in bp2['boxes']:
        box.set_facecolor(color_laser)

    # Add labels and titles
    ax.set_xlabel('Laser Event Phase (in radians)', fontsize=14)
    ax.set_ylabel('Breath Duration (in seconds)', fontsize=14)
    ax.set_title('Breath Duration vs Phase of Laser Inhibition (1ms)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Add legend
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Preceding Breath Durations', 'Laser Breath Durations'], fontsize=12)

    plt.show()

# Function to plot laser and breath events
def plot_laser_and_breath_events(example_laser_event_start, example_laser_breath_insp_start, example_next_insp_idx, filtered_breathing_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(example_laser_breath_insp_start, example_next_insp_idx), filtered_breathing_signal[example_laser_breath_insp_start:example_next_insp_idx], label="Filtered Breathing Signal")
    plt.axvline(x=example_laser_event_start, color='m', linestyle='--', label="Laser Event Start")
    plt.axvline(x=example_next_insp_idx, color='r', linestyle='--', label="Next Inspiration Start")
    plt.hlines(y=filtered_breathing_signal[example_laser_breath_insp_start], xmin=example_laser_breath_insp_start, xmax=example_next_insp_idx, colors='r', linestyles='-', label="Time to Next Inspiration")
    
    plt.title("Corrected: Example Laser Event and Time to Next Inspiration")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()