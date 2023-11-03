__all__ = ['plot_cycle_lengths_vs_phase', 'plot_binned_laser_phases_vs_breath_length', 'plot_breath_phases']

#import stuff
import os
import pandas as pd 
import numpy as np
from config import *
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
from pleth_breathing_analysis.voltageSignalProcessing import *
import seaborn as sns

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
    plt.show(block=False)

def plot_binned_laser_phases_vs_breath_length(preceding_breath_durations=None, laser_breath_durations=None, laser_event_phase_values=None, num_bins=10, exclude_data=False):
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

    # Filter out laser event phase values that are more than 2 standard deviations from the mean
    mean_phase = np.mean(laser_event_phase_values)
    std_phase = np.std(laser_event_phase_values)
    filtered_phase_values = []
    filtered_preceding_durations = []
    filtered_laser_durations = []

    num_excluded = 0
    for i, phase in enumerate(laser_event_phase_values):
        if phase <= 2 * np.pi:
            filtered_phase_values.append(phase)
            filtered_preceding_durations.append(preceding_breath_durations[i])
            filtered_laser_durations.append(laser_breath_durations[i])
        else:
            num_excluded += 1

    # Report the number of excluded phase values
    print(f"Number of phase values excluded: {num_excluded}")

    # Group data into bins
    min_phase = 0
    max_phase = 2 * np.pi
    bin_edges = np.linspace(min_phase, max_phase, num_bins + 1)
    bin_indices = np.digitize(filtered_phase_values, bin_edges)

    bins = {}
    for i, bin_idx in enumerate(bin_indices):
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append((filtered_preceding_durations[i], filtered_laser_durations[i]))

    # Optionally make sure each bin has an equal number of data points
    if exclude_data:
        min_bin_size = min(len(b) for b in bins.values())
        total_removed_points = 0

        for bin_idx in bins.keys():
            num_removed_points = len(bins[bin_idx]) - min_bin_size
            total_removed_points += num_removed_points
            random.shuffle(bins[bin_idx])
            bins[bin_idx] = bins[bin_idx][:min_bin_size]

        print(f"Total number of data points removed for equal bin sizes: {total_removed_points}")

    # Create the box and whisker plot
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
    preceding_data = [[preceding for preceding, _ in bins.get(i+1, [])] for i in range(num_bins)]
    laser_data = [[laser for _, laser in bins.get(i+1, [])] for i in range(num_bins)]
    bin_labels = [f"{bin_centers[i]:.2f}" for i in range(num_bins)]

    fig, ax = plt.subplots()
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_preceding = default_colors[0]
    color_laser = default_colors[1]

    bp1 = ax.boxplot(preceding_data, positions=bin_centers, widths=np.diff(bin_edges), patch_artist=True, labels=bin_labels, showfliers=False)
    bp2 = ax.boxplot(laser_data, positions=bin_centers, widths=np.diff(bin_edges), patch_artist=True, labels=bin_labels, showfliers=False)

    for box in bp1['boxes']:
        box.set_facecolor(color_preceding)
    for box in bp2['boxes']:
        box.set_facecolor(color_laser)

    ax.set_xlabel('Laser Event Phase (in radians)', fontsize=14)
    ax.set_ylabel('Breath Duration (in seconds)', fontsize=14)
    ax.set_title('CONTROL (10ms)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Preceding Breath Durations', 'Laser Breath Durations'], fontsize=12)

    plt.show()
    return num_excluded

def plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=''):
    preceding_full_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[1] <= laser_event_start][-1]
    preceding_expiration_start = expiration_indices[expiration_indices > preceding_full_breath_cycle[0]][0]
    preceding_accurate_breath_phases = calculate_breath_phases(preceding_full_breath_cycle[0], preceding_expiration_start, preceding_full_breath_cycle[1])

    # Extend and match the breath phase
    extended_phases, laser_event_phase = extend_and_match_breath_phase(preceding_accurate_breath_phases, preceding_full_breath_cycle[0], preceding_full_breath_cycle[1], laser_event_start)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_breathing_signal[preceding_full_breath_cycle[0]:laser_event_start], label="Filtered Breathing Signal")
    plt.axvline(x=laser_event_start - preceding_full_breath_cycle[0], color='m', linestyle='--', label=f"Laser Event Start")
    plt.twinx()
    plt.plot(extended_phases, 'g--', label="Extended Breath Phases (in radians)")
    plt.ylabel("Phase (radians)")
    plt.legend(loc="upper right")
    plt.title(f"Extended Breath Phase for Laser Event {title_suffix}")
    #plt.show()

    return laser_event_phase