from wav.wav_signal_processing import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from config import *

# Main script to process all WAV files in a folder and plot the aggregated results
if __name__ == "__main__":
    folder_path = FOLDER_PATH
    all_phases = []
    all_times_after = []
    all_times_before = []
    event_duration_ms = EVENT_DURATION_MS
    num_breaths_before = NUM_BREATHS_BEFORE

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            phases, times_after, times_before = process_wav_file(file_path, event_duration_ms, num_breaths_before)
            all_phases.extend(phases)
            all_times_after.extend(times_after)
            all_times_before.extend(times_before)

    # Convert lists to numpy arrays for numerical operations
    all_phases = np.array(all_phases)
    all_times_after = np.array(all_times_after)
    all_times_before = np.array(all_times_before)

    # Sort data based on phases for plotting
    sorted_indices = np.argsort(all_phases)
    sorted_phases = all_phases[sorted_indices]
    sorted_times_after = all_times_after[sorted_indices]
    sorted_times_before = all_times_before[sorted_indices]

    # Perform linear regression to fit lines
    slope_after, intercept_after, _, _, _ = linregress(sorted_phases, sorted_times_after)
    slope_before, intercept_before, _, _, _ = linregress(sorted_phases, sorted_times_before)

    # Generate y-values based on the linear regression model
    line_after = slope_after * sorted_phases + intercept_after
    line_before = slope_before * sorted_phases + intercept_before

    # Calculate the variance for shading
    var_after = np.var(sorted_times_after)
    var_before = np.var(sorted_times_before)

    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.scatter(sorted_phases, sorted_times_after, label='Data Points After Laser', color='r', alpha=0.6)
    plt.scatter(sorted_phases, sorted_times_before, label='Data Points Before Laser', color='b', alpha=0.6)
    plt.plot(sorted_phases, line_after, label='Regression Line After Laser', color='r')
    plt.fill_between(sorted_phases, line_after - var_after, line_after + var_after, color='r', alpha=0.2)
    plt.plot(sorted_phases, line_before, label='Regression Line Before Laser', color='b')
    plt.fill_between(sorted_phases, line_before - var_before, line_before + var_before, color='b', alpha=0.2)
    plt.title('Aggregated Data Across Multiple Recordings')
    plt.xlabel('Phase at Laser Event (radians)')
    plt.ylabel('Time to Next Inspiration (s)')
    plt.legend()
    # Save the plot as a file
    plot_file_name = f"{os.path.basename(folder_path)}_aggregated_plot.png"
    plot_file_path = os.path.join(folder_path, plot_file_name)
    plt.savefig(plot_file_path)

    # Show the plot
    plt.show()
