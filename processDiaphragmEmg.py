# Imports
from pleth_breathing_analysis.voltageSignalProcessing import *
from pleth_breathing_analysis.plots import *
from scipy.io import wavfile
from config import *
import random

if __name__ == "__main__":
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(r'C:\Users\Beatriz\data\nk1roprm1-chr2-co1\572_03_oprm1nklr_optotag_pbc.wav')


import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# Assign Channels
breathing_channel = audio_data[:, 1]
laser_channel = audio_data[:, 0]
diaphragm_channel = audio_data[:, 2]

# Filter the breathing channel
filtered_breathing_signal = bandpass_filter(breathing_channel, 0.5, 40, sample_rate)

# Plot the signals for visual inspection
plt.figure(figsize=(15, 5))

# Plot breathing channel
plt.subplot(3, 1, 1)
plt.plot(filtered_breathing_signal)
plt.title('Breathing')

# Plot laser channel
plt.subplot(3, 1, 2)
plt.plot(laser_channel)
plt.title('Laser')

# Plot diaphragm channel
plt.subplot(3, 1, 3)
plt.plot(diaphragm_channel)
plt.title('Diaphragm EMG')

plt.tight_layout()
plt.show()

# Define the moving average function
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

# Rectify the EMG signal
rectified_emg = np.abs(diaphragm_channel)

# Filter out high amplitude noise that may be ECG or other artifacts
amplitude_threshold = 5000
thresholded_emg = np.where(rectified_emg > amplitude_threshold, 0, rectified_emg)

# Calculate the window size for the moving average based on the sample rate and the desired window duration
window_duration = 0.10  # 100ms window
window_size = int(window_duration * sample_rate)

# Apply the moving average filter
smoothed_emg = moving_average(thresholded_emg, window_size)

# Set amplitudes below 500 to zero
smoothed_emg[smoothed_emg < 500] = 500

# Compute the average of the smoothed signal
average_emg = np.mean(smoothed_emg)




# Plot the original, rectified, thresholded, and smoothed EMG signals for comparison
plt.figure(figsize=(15, 8))

# Plot original EMG signal
plt.subplot(4, 1, 1)
plt.plot(diaphragm_channel)
plt.title('Original Preprocessed EMG Signal')

# Plot rectified EMG signal
plt.subplot(4, 1, 2)
plt.plot(rectified_emg)
plt.title('Rectified EMG Signal')

# Plot thresholded EMG signal
plt.subplot(4, 1, 3)
plt.plot(thresholded_emg)
plt.title('Thresholded EMG Signal')

# Plot smoothed EMG signal
plt.subplot(4, 1, 4)
plt.plot(smoothed_emg)
plt.title('Smoothed EMG Signal')

plt.tight_layout()
plt.show()



# Define the time axis (assuming 1000 Hz sampling rate)
time = np.arange(0, len(diaphragm_channel)) / 1000.0  # Change the 1000 to your actual sampling rate

# Plot the original, rectified, thresholded, and smoothed EMG signals for comparison
plt.figure(figsize=(15, 8))

# Plot original EMG signal for the first 5 seconds
plt.subplot(4, 1, 1)
plt.plot(time[:5000], diaphragm_channel[:5000])
plt.title('Original Preprocessed EMG Signal')

# Plot rectified EMG signal for the first 5 seconds
plt.subplot(4, 1, 2)
plt.plot(time[:5000], rectified_emg[:5000])
plt.title('Rectified EMG Signal')

# Plot thresholded EMG signal for the first 5 seconds
plt.subplot(4, 1, 3)
plt.plot(time[:5000], thresholded_emg[:5000])
plt.title('TWO Thresholded EMG Signal')

# Plot smoothed EMG signal for the first 5 seconds
plt.subplot(4, 1, 4)
plt.plot(time[:5000], smoothed_emg[:5000])
plt.title('Smoothed EMG Signal')

plt.tight_layout()
plt.show()


# Define a threshold for laser channel
laser_threshold = 4000

# Find the indices where laser channel crosses the threshold
above_threshold_indices = np.where(laser_channel > laser_threshold)[0]

# Create plots for 3-second windows around each crossing
window_size = 15 * sample_rate  # 3 seconds in samples

plt.figure(figsize=(15, 5))

# Keep track of the previous index to ensure only one event is included
prev_index = None

for idx in above_threshold_indices:
    if prev_index is None or idx - prev_index > window_size:
        start_idx = max(0, idx - window_size // 2)
        end_idx = min(len(audio_data), idx + window_size // 2)
        
        # Plot breathing channel
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(start_idx, end_idx) / 1000, filtered_breathing_signal[start_idx:end_idx])
        plt.title('Breathing')

        # Plot laser channel
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(start_idx, end_idx) / 1000, laser_channel[start_idx:end_idx])
        plt.title('Laser')

        # Plot diaphragm channel
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(start_idx, end_idx) / 1000, smoothed_emg[start_idx:end_idx])
        plt.title('Diaphragm EMG')
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.show()
        # Update the previous index
        prev_index = idx