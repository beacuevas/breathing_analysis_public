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

def process_wav_file(file_path, sample_rate=1000):
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(file_path)
        
    # Assign Channels
    breathing_channel = audio_data[:, 0]
    laser_channel = audio_data[:, 1]  

    # Filter the breathing channel
    filtered_breathing_signal = bandpass_filter(breathing_channel, 2, 30, sample_rate)
        
    # Find inspirations in the filtered breathing signal
    inspiration_indices = find_inspirations(filtered_breathing_signal, min_duration=0.03, min_amplitude=20)
        
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
    laser_breath_phases = []
    time_to_next_inspirations = []

    for event_start, event_end in filtered_laser_events:
        # Find the full breath cycle that contains the laser event
        containing_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]][0]
        
        # Find the expiration index that corresponds to the inspiration start of the containing breath cycle
        corresponding_expiration_start = expiration_indices[expiration_indices > containing_breath_cycle[0]][0]
        
        # Calculate the accurate breath phases for the containing breath cycle
        laser_breath_phase = calculate_breath_phases(containing_breath_cycle[0], corresponding_expiration_start, containing_breath_cycle[1])
        
        # Calculate time to next inspiration from the start of the containing breath cycle
        time_to_next_insp, laser_breath_insp_start, next_insp_idx = time_to_next_inspiration_from_insp_start(event_start, inspiration_indices, sample_rate)
        
        laser_breath_phases.append(laser_breath_phase)
        time_to_next_inspirations.append(time_to_next_insp)

    # Select two random events from filtered_laser_events and plot
    random_events = random.sample(filtered_laser_events, 2)
    for event_start, event_end in random_events:
        # Find the breath cycle, expiration index, and other parameters just like before
        containing_breath_cycle = [cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]][0]
        corresponding_expiration_start = expiration_indices[expiration_indices > containing_breath_cycle[0]][0]
        time_to_next_insp, laser_breath_insp_start, next_insp_idx = time_to_next_inspiration_from_insp_start(event_start, inspiration_indices, sample_rate)
        # Plot this event
        plot_laser_and_breath_events(event_start, laser_breath_insp_start, next_insp_idx, filtered_breathing_signal)


    # Identify the breath that contains each laser event- tuples that define the start and end of the breath that contains each laser event. 
    containing_breath_cycles_for_laser_events = [next(cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]) 
                                                for event_start, _ in filtered_laser_events]

    # Test for synchrony between lists so that this next analysis is accurate. These tests MUST pass for the sake of accurate analysis. 
    # Perform statistical direct validation
    is_valid, message = statistical_direct_validation(laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events)
    print(f"Statistical Direct Validation: {message}")

    # Perform temporal consistency check
    is_consistent, message = temporal_consistency_check(filtered_laser_events, containing_breath_cycles_for_laser_events)
    print(f"Temporal Consistency Check: {message}")

    laser_event_phase_values = [laser_breath_phase[(event_start - cycle[0]) % len(laser_breath_phase)] 
                                for laser_breath_phase, (event_start, _), cycle 
                                in zip(laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events)]

    # Get the length (in time) of the breaths 
    preceding_breath_durations, laser_breath_durations = calculate_breath_durations(containing_breath_cycles_for_laser_events, full_breath_cycles, sample_rate=1000, num_preceding_breaths=1)

    plot_cycle_lengths_vs_phase(preceding_breath_durations, laser_breath_durations, laser_event_phase_values)

    create_box_and_whisker_plot(preceding_breath_durations, laser_breath_durations, laser_event_phase_values, num_bins=10, exclude_data=False)

    return {
        'preceding_breath_durations': preceding_breath_durations,
        'laser_breath_durations': laser_breath_durations,
        'laser_event_phase_values': laser_event_phase_values
    }

def process_wav_folder(folder_path, sample_rate=1000, all_preceding_durations=None, all_laser_durations=None, all_laser_phases=None):
    if all_preceding_durations is None:
        all_preceding_durations = []
    if all_laser_durations is None:
        all_laser_durations = []
    if all_laser_phases is None:
        all_laser_phases = []
    
    total_files = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
    print(f"Total files to process: {total_files}")

    for idx, file_name in enumerate(os.listdir(folder_path)):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file {idx + 1}/{total_files}: {file_path}...")
            
            result = process_wav_file(file_path, sample_rate)  

            all_preceding_durations.extend(result['preceding_breath_durations'])
            all_laser_durations.extend(result['laser_breath_durations'])
            all_laser_phases.extend(result['laser_event_phase_values'])

    # Debug same data issue
    print("Unique preceding durations:", len(set(all_preceding_durations)))
    print("Unique laser durations:", len(set(all_laser_durations)))    
    # Plot aggregated data
    plot_cycle_lengths_vs_phase(all_preceding_durations, all_laser_durations, all_laser_phases)
    create_box_and_whisker_plot(all_preceding_durations, all_laser_durations, all_laser_phases, num_bins=15, exclude_data=False)

if __name__ == "__main__":
    folder_path = FOLDER_PATH
    process_wav_folder(folder_path)
