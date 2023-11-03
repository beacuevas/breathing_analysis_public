import os
from pleth_breathing_analysis.voltageSignalProcessing import *
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import seaborn as sns
from config import *


if __name__ == "__main__":
    folder_path = FOLDER_PATH  
    event_duration_ms = EVENT_DURATION_MS
    dfs = []  # Container to hold individual laser_timing_df DataFrames
    num_bins = NUM_PHASE_BINS
    num_breaths_before = NUM_BREATHS_BEFORE

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            phases, times_after, times_before = process_wav_file(file_path, event_duration_ms, num_breaths_before)

            laser_timing_df = pd.DataFrame({
                'Breath_Phase_At_Laser': phases,
                'Time_to_Next_Insp_After_Laser_From_Insp_Start': times_after,
                'Avg_Time_to_Next_Insp_Before_Laser': times_before
            })

            # Bin the data evenly across phase (from Script #1)
            bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)
            bin_labels = np.arange(num_bins)
            laser_timing_df['Phase_Bin'] = pd.cut(laser_timing_df['Breath_Phase_At_Laser'], bins=bin_edges, labels=bin_labels)

            # Count the number of samples in each bin and find the bin with the least number of samples
            bin_counts = laser_timing_df['Phase_Bin'].value_counts().sort_index()
            min_bin_count = bin_counts.min()

            # Randomly select 'min_bin_count' samples from each bin
            equal_samples_df = laser_timing_df.groupby('Phase_Bin').apply(lambda x: x.sample(min_bin_count)).reset_index(drop=True)

            dfs.append(equal_samples_df)  # Add this DataFrame to the container

    # Concatenate all DataFrames into one
    aggregated_df = pd.concat(dfs, ignore_index=True)

    # Create the boxplots and violin plots (from Script #1)
    bin_labels_pi_range = [f"{start/np.pi:.2f}π-{end/np.pi:.2f}π" for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    plt.figure(figsize=(16, 7))
    sns.boxplot(x='Phase_Bin', y='value', hue='variable', data=pd.melt(aggregated_df, id_vars=['Phase_Bin'], value_vars=['Time_to_Next_Insp_After_Laser_From_Insp_Start', 'Avg_Time_to_Next_Insp_Before_Laser']))
    plt.title('Boxplots of Time to Next Inspiration by Even Breath Phase Bins')
    plt.xlabel('Breath Phase Range')
    plt.xticks(ticks=range(len(bin_labels_pi_range)), labels=bin_labels_pi_range, rotation=45)
    plt.ylabel('Time to Next Inspiration (s)')
    plt.legend(title='Time Reference', labels=['After Laser', 'Before Laser'])
    plt.show()

    plt.figure(figsize=(16, 7))
    sns.violinplot(x='Phase_Bin', y='value', hue='variable', data=pd.melt(aggregated_df, id_vars=['Phase_Bin'], value_vars=['Time_to_Next_Insp_After_Laser_From_Insp_Start', 'Avg_Time_to_Next_Insp_Before_Laser']), inner='quartile')
    plt.title('Violin Plots of Time to Next Inspiration by Even Breath Phase Bins')
    plt.xlabel('Breath Phase Range')
    plt.xticks(ticks=range(len(bin_labels_pi_range)), labels=bin_labels_pi_range, rotation=45)
    plt.ylabel('Time to Next Inspiration (s)')
    plt.legend(title='Time Reference', labels=['After Laser', 'Before Laser'])
    plt.show()
