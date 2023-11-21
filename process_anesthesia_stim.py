# Imports
from pleth_breathing_analysis.voltageSignalProcessing import *
from pleth_breathing_analysis.plots import *
from scipy.io import wavfile
from config import *
import random
import numpy as np
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt
sample_rate = 30000
# Function to read a binary DAT file and return its content as a numpy array
def read_dat_file(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# File paths for the DAT files
file_path_analog_1 = r'C:\Users\Beatriz\data\nk1roprm1-stgtacr-co1\anesthetized\588_30_nk1roprm1_stgtacr2_co1_anesthetized_231121_130133\board-ANALOG-IN-1.dat'  # Replace with your file path
file_path_analog_2 = r'C:\Users\Beatriz\data\nk1roprm1-stgtacr-co1\anesthetized\588_30_nk1roprm1_stgtacr2_co1_anesthetized_231121_130133\board-ANALOG-IN-2.dat'  # Replace with your file path

# Load the data from both files
breathing_channel = read_dat_file(file_path_analog_2)
laser_channel = read_dat_file(file_path_analog_1)

# # Filter the breathing channel
filtered_breathing_signal = bandpass_filter(breathing_channel, 0.5, 40, 30000)

# Helper function to find laser onset times
def find_laser_onsets(laser_signal, threshold, sample_rate):
    # Find the indices where laser channel crosses the threshold
    above_threshold_indices = np.where(laser_signal > threshold)[0]
    # Debounce the onsets by taking only the first sample in a contiguous above-threshold region
    onsets = [above_threshold_indices[0]]
    for i in range(1, len(above_threshold_indices)):
        if above_threshold_indices[i] - above_threshold_indices[i - 1] > sample_rate:
            onsets.append(above_threshold_indices[i])
    return onsets

# Constants
LASER_THRESHOLD = 1
BEFORE_LASER_SEC = 5
AFTER_LASER_SEC = 30

# Sample rate and audio data for demonstration purposes
sample_rate = 10000  # 1000 samples per second
duration_sec = 120  # 2 minutes

# Find laser onset times
laser_onsets = find_laser_onsets(laser_channel, LASER_THRESHOLD, sample_rate)

# Calculate number of samples to include before and after laser onset
samples_before = BEFORE_LASER_SEC * sample_rate
samples_after = AFTER_LASER_SEC * sample_rate

# Set up the figure and axes for the subplots
num_onsets = len(laser_onsets)
fig, axes = plt.subplots(num_onsets, figsize=(15, 3 * num_onsets))

# If there's only one onset, wrap the axes variable in a list so that we can iterate over it
if num_onsets == 1:
    axes = [axes]

# Loop through each laser onset and plot in a separate subplot
for i, onset in enumerate(laser_onsets):
    # Define the window around the laser onset
    start_idx = max(0, onset - samples_before)
    end_idx = min(len(breathing_channel), onset + samples_after)
    
    # Time vector for x-axis, offset to zero at the laser onset
    time_vector = (np.arange(start_idx, end_idx) - onset) / sample_rate
    
    # Plot the segment of the breathing signal
    axes[i].plot(time_vector, breathing_channel[start_idx:end_idx], label='Breathing Signal')
    
    # Highlight the laser period
    axes[i].axvspan(0, 3, color='red', alpha=0.3, label='Laser Period' if i == 0 else "")
    
    
    # Set titles and labels
    axes[i].set_title(f'Laser onset {i+1}')
    axes[i].set_ylabel('Amplitude')
    
    # Only add the legend to the first subplot
    if i == 0:
        axes[i].legend(loc='upper right')

# Set common xlabel for the last subplot
axes[-1].set_xlabel('Time (seconds)')

fig.tight_layout()
plt.show()