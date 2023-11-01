
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import seaborn as sns
from wav.wav_signal_processing import *
from config import *

if __name__ == "__main__":
    event_duration_ms = EVENT_DURATION_MS
    file_path = FILE_PATH
    num_breaths_before = NUM_BREATHS_BEFORE
    phases, times_after, times_before = process_wav_file(file_path, event_duration_ms, num_breaths_before)
    corrected_filtered_laser_event_phases =  phases

    # Create a DataFrame to store the calculated values
    laser_timing_df = pd.DataFrame({
        'Time_to_Next_Insp_After_Laser_From_Insp_Start': times_after,
        'Avg_Time_to_Next_Insp_Before_Laser': times_before
    })

    # Sort the data for consistent plotting
    sorted_indices = np.argsort(corrected_filtered_laser_event_phases)
    sorted_phases = np.array(corrected_filtered_laser_event_phases)[sorted_indices]
    sorted_time_after = np.array(laser_timing_df['Time_to_Next_Insp_After_Laser_From_Insp_Start'])[sorted_indices]
    sorted_time_before = np.array(laser_timing_df['Avg_Time_to_Next_Insp_Before_Laser'])[sorted_indices]

    # Function to perform linear regression and plot the line with variance shading
    def plot_linear_regression_with_variance(x, y, color, label):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * np.array(x) + intercept
        plt.plot(x, line, color=color, label=f"{label} (R2 = {r_value**2:.2f})")

        # Calculate the variance and plot it as shading
        variance = np.var(y)
        plt.fill_between(x, line - np.sqrt(variance), line + np.sqrt(variance), color=color, alpha=0.2)

    # Plotting the final visualization with linear regression lines and variance shading
    plt.figure(figsize=(15, 8))

    # Scatter plot for individual points
    plt.scatter(sorted_phases, sorted_time_after, label='Time to Next Breath After Laser', color='r', alpha=0.6)
    plt.scatter(sorted_phases, sorted_time_before, label='Avg. Time to Next Breath Before Laser', color='b', alpha=0.6)

    # Linear regression with variance shading for 'After Laser'
    plot_linear_regression_with_variance(sorted_phases, sorted_time_after, 'darkred', 'Linear Fit After Laser')

    # Linear regression with variance shading for 'Before Laser'
    plot_linear_regression_with_variance(sorted_phases, sorted_time_before, 'darkblue', 'Linear Fit Before Laser')

    # Additional plot settings
    plt.title('Time to Next Inspiration vs. Phase of Breathing')
    plt.xlabel('Phase at Laser Event (radians)')
    plt.ylabel('Time to Next Inspiration (s)')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    plt.legend()
    plt.grid(True)
    plt.show()
