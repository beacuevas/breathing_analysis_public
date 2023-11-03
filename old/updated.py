# Imports
from pleth_breathing_analysis.voltageSignalProcessing import *
from pleth_breathing_analysis.plots import *
from scipy.io import wavfile
from config import *
import random

if __name__ == "__main__":
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(FILE_PATH)
        
    # Assign Channels
    breathing_channel = audio_data[:, 0]
    laser_channel = audio_data[:, 1]  

    # Filter the breathing channel
    filtered_breathing_signal = bandpass_filter(breathing_channel, 2, 40, sample_rate)
        
    # Find inspirations in the filtered breathing signal
    inspiration_indices = find_inspirations(filtered_breathing_signal, min_duration=0.02, min_amplitude=25)    
    # Find expirations and full breath cycles
    expiration_indices = find_expirations(filtered_breathing_signal, inspiration_indices)
    full_breath_cycles = find_full_breaths(inspiration_indices, expiration_indices)

    # Find laser events for plotting
    laser_events = find_laser_events(laser_channel, 1)

    # Process all laser events
    all_laser_event_phases = find_laser_event_phases(laser_events, full_breath_cycles, expiration_indices, filtered_breathing_signal)

    # Choose 4 laser events at random to plot
    random_laser_event_indices = random.sample(range(len(laser_events)), 4)
    for i, idx in enumerate(random_laser_event_indices, start=1):
        laser_event_start = laser_events[idx][0]
        plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=f"Random #{i}")

    # Identify the breath that contains each laser event- tuples that define the start and end of the breath that contains each laser event. 
    # Works by  iterating through each laser event and finding the breath cycle it falls within.  
    containing_breath_cycles_for_laser_events = [next(cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]) 
                                                for event_start, _ in laser_events]
            # If there are more than one breaths that meet this criteria then only the first will be chosen. Make a test that assures that this will not be the case.
            # This DOES NOT WORK for laser events that are longer than a breath. 

    # Get the length (in time) of the breaths 
    preceding_breath_durations, laser_breath_durations = calculate_breath_durations(containing_breath_cycles_for_laser_events, full_breath_cycles, sample_rate=1000, num_preceding_breaths=1)

    plot_cycle_lengths_vs_phase(preceding_breath_durations, laser_breath_durations, all_laser_event_phases)

    # This plotting method shuffles tuples of preceding durations and laser durations in order to not lose the control and experimental pairing when 
    # making the bins the same size. Due to the Random shuffling, it could mean that plots are slightly different every time. 
    num_excluded = plot_binned_laser_phases_vs_breath_length(preceding_breath_durations, laser_breath_durations, all_laser_event_phases, num_bins=15, exclude_data=False)

    # Plot all breaths that got excluded 
    #Sorting the indices based on the values in descending order
    sorted_indices = sorted(range(len(all_laser_event_phases)), key=lambda i: all_laser_event_phases[i], reverse=True)
    # Getting the indices of the 10 longest durations
    longest_laser_event_indices = sorted_indices[:num_excluded]

    for i, idx in enumerate(longest_laser_event_indices, start=1):
        event_start =[]
        event_start = laser_events[idx][0]
        plot_breath_phases(event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=f"Longest #{i}")