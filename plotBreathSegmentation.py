import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import seaborn as sns
from wav.wav_signal_processing import *
from config import *

def plot_random_full_breaths(file_path, num_breaths=5):
    """
    Generate a plot of random full breath cycles from a given WAV file.
    
    Parameters:
    - file_path (str): The path to the WAV file.
    - num_breaths (int): The number of random full breath cycles to plot.
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(FILE_PATH)
    
    # Assuming the first channel is the breathing channel
    breathing_channel = audio_data[:, 0]
    
    # Filter the breathing channel
    filtered_breathing_signal = bandpass_filter(breathing_channel, 2, 15, sample_rate)
    
    # Find inspirations in the filtered breathing signal
    inspiration_indices = find_inspirations(filtered_breathing_signal)
    
    # Find expirations and full breath cycles
    expiration_indices = find_expirations(filtered_breathing_signal, inspiration_indices)
    full_breath_cycles = find_full_breaths(inspiration_indices, expiration_indices)
    
    # Randomly select full breath cycles to plot
    random_full_breath_cycles = random.sample(full_breath_cycles, num_breaths)
    
    # Prepare the plot
    plt.figure(figsize=(14, 8))
    
    for start_idx, end_idx in random_full_breath_cycles:
        # Extract the individual full breath cycle (from inspiration to the next inspiration)
        individual_breath_cycle = filtered_breathing_signal[start_idx:end_idx+1]
        
        # Create an array for the x-axis (time index)
        time_index = np.arange(0, len(individual_breath_cycle))
        
        # Find the corresponding expiration index within this full breath cycle
        exp_idx_relative = expiration_indices[(expiration_indices > start_idx) & (expiration_indices < end_idx)][0] - start_idx
        
        # Plot the individual full breath cycle
        plt.plot(time_index, individual_breath_cycle, label=f'Full Breath Cycle starting at {start_idx}')
        
        # Shade the expiration phase (from the end of inspiration to the start of next inspiration)
        plt.fill_between(time_index, individual_breath_cycle, where=(time_index >= exp_idx_relative), color='r', alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Time Index (relative to each breath cycle)')
    plt.ylabel('Amplitude')
    plt.title('Overlay of Random Individual Full Breath Cycles')
    plt.legend()
    
    plt.show()

# Generate the required plot from the provided WAV file
plot_random_full_breaths(FILE_PATH, 100)
