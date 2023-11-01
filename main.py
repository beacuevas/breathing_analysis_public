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
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')


# Read the WAV file
sample_rate, audio_data = wavfile.read(FILE_PATH)
    
# Assign Channels
breathing_channel = audio_data[:, 0]
laser_channel = audio_data[:, 1]  

# Filter the breathing channel
filtered_breathing_signal = bandpass_filter(breathing_channel, 2, 15, sample_rate)
    
# Find inspirations in the filtered breathing signal
inspiration_indices = find_inspirations(filtered_breathing_signal)
    
# Find expirations and full breath cycles
expiration_indices = find_expirations(filtered_breathing_signal, inspiration_indices)
full_breath_cycles = find_full_breaths(inspiration_indices, expiration_indices)

# Find laser events for plotting
filtered_laser_events = find_laser_events(laser_channel, 1)

# Plot accurate breath phases for two example laser events
first_two_laser_event_starts = [event[0] for event in filtered_laser_events[:2]]
for i, laser_event_start in enumerate(first_two_laser_event_starts, start=1):
    plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=f"#{i}")

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

example_laser_event_start = filtered_laser_events[0][0]
example_time_to_next_insp, example_laser_breath_insp_start, example_next_insp_idx = time_to_next_inspiration_from_insp_start(example_laser_event_start, inspiration_indices, sample_rate)
plot_laser_and_breath_events(example_laser_event_start, example_laser_breath_insp_start, example_next_insp_idx, filtered_breathing_signal)

# Identify the breath that contains each laser event- tuples that define the start and end of the breath that contains each laser event. 
# Works by  iterating through each laser event and finding the breath cycle it falls within.  
containing_breath_cycles_for_laser_events = [next(cycle for cycle in full_breath_cycles if cycle[0] <= event_start <= cycle[1]) 
                                             for event_start, _ in filtered_laser_events]
        # If there are more than one breaths that meet this criteria then only the first will be chosen. Make a test that assures that this will not be the case.
        # This DOES NOT WORK for laser events that are longer than a breath. 


# Test for synchrony between lists so that this next analysis is accurate. These tests MUST pass for the sake of accurate analysis. 
# Perform statistical direct validation
is_valid, message = statistical_direct_validation(laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events)
print(f"Statistical Direct Validation: {message}")

# Perform temporal consistency check
is_consistent, message = temporal_consistency_check(filtered_laser_events, containing_breath_cycles_for_laser_events)
print(f"Temporal Consistency Check: {message}")

# Extract the breath phase at the point where each laser event occurs within the breath cycle, using modulo indexing
# Zip simultaneously iterates over 3 lists: laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events. During each iteration, 
# the script has access to laser_breath_phase (an array of breath phase values for the breath cycle containing the current laser event); 
# event_start (which is the start index of the laser event); and cycle (a tuple (inspiration_start, inspiration_end) indicating the 
# start and end indices of the breath cycle containing the laser event). Then,  the expression (event_start - cycle[0]) % len(laser_breath_phase) is used
# to calculate the index within the laser_breath_phase array that corresponds to the start of the laser event. The script then uses this index to extract 
# the corresponding breath phase value from laser_breath_phase. The extracted breath phase value is added to the list laser_event_phase_values.
laser_event_phase_values = [laser_breath_phase[(event_start - cycle[0]) % len(laser_breath_phase)] 
                            for laser_breath_phase, (event_start, _), cycle 
                            in zip(laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events)]
        # DOES NOT WORK IF Synchronization IS NOT TRUE: It assumes that the three lists (laser_breath_phases, filtered_laser_events, containing_breath_cycles_for_laser_events) are in sync,
        #  meaning each set of elements across these lists corresponds to the same laser event. --> Make test for synchronization a. test a subset directly in a random subset 
        #  and also test for temporal consistency

# Get the length (in time) of the breaths 
preceding_breath_durations, laser_breath_durations = calculate_breath_durations(containing_breath_cycles_for_laser_events, full_breath_cycles, sample_rate=1000, num_preceding_breaths=1)

plot_cycle_lengths_vs_phase(preceding_breath_durations, laser_breath_durations, laser_event_phase_values)

# This plotting method shuffles tuples of preceding durations and laser durations in order to not lose the control and experimental pairing when 
# making the bins the same size. Due to the Random shuffling, it could mean that plots are slightly different every time. 
create_box_and_whisker_plot(preceding_breath_durations, laser_breath_durations, laser_event_phase_values, num_bins=10, exclude_data=False)
