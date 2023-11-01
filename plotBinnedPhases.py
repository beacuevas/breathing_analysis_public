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
    file_path = FILE_PATH
    event_duration_ms = EVENT_DURATION_MS  
    phases, times_after, times_before = process_wav_file(file_path, event_duration_ms)
    corrected_filtered_laser_event_phases =  phases

    # Create a DataFrame to store the calculated values
    laser_timing_df = pd.DataFrame({
        'Breath_Phase_At_Laser': corrected_filtered_laser_event_phases,
        'Time_to_Next_Insp_After_Laser_From_Insp_Start': times_after,
        'Avg_Time_to_Next_Insp_Before_Laser': times_before
    })
    # Step 1: Segment the breath phase into bins evenly across phase (0 to 2π)
    num_bins = NUM_PHASE_BINS 

    # Define the bin edges and bin labels
    bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)
    bin_labels = np.arange(num_bins)

    # Assign bin numbers to each row based on the breath phase
    laser_timing_df['Phase_Bin'] = pd.cut(laser_timing_df['Breath_Phase_At_Laser'], bins=bin_edges, labels=bin_labels)

    # Count the number of samples in each bin and find the bin with the least number of samples
    bin_counts = laser_timing_df['Phase_Bin'].value_counts().sort_index()
    min_bin_count = bin_counts.min()

    # Randomly select 'min_bin_count' samples from each bin to make the number of samples in each bin equal
    equal_samples_df = laser_timing_df.groupby('Phase_Bin').apply(lambda x: x.sample(min_bin_count)).reset_index(drop=True)

    # Update the x-axis labels to indicate the phase range (start-end) that each bin encompasses
    bin_labels_pi_range = [f"{start/np.pi:.2f}π-{end/np.pi:.2f}π" for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    # Create the visualizations with new x-axis labels indicating bin range

    # Updated Boxplots grouped by Phase Bins
    plt.figure(figsize=(16, 7))
    sns.boxplot(x='Phase_Bin', y='value', hue='variable', data=pd.melt(equal_samples_df, id_vars=['Phase_Bin'], value_vars=['Time_to_Next_Insp_After_Laser_From_Insp_Start', 'Avg_Time_to_Next_Insp_Before_Laser']))
    plt.title('Boxplots of Time to Next Inspiration by Even Breath Phase Bins')
    plt.xlabel('Breath Phase Range')
    plt.xticks(ticks=range(len(bin_labels_pi_range)), labels=bin_labels_pi_range, rotation=45)
    plt.ylabel('Time to Next Inspiration (s)')
    plt.legend(title='Time Reference', labels=['After Laser', 'Before Laser'])
    plt.show()

    # Updated Violin Plots with Phase Bins
    plt.figure(figsize=(16, 7))
    sns.violinplot(x='Phase_Bin', y='value', hue='variable', data=pd.melt(equal_samples_df, id_vars=['Phase_Bin'], value_vars=['Time_to_Next_Insp_After_Laser_From_Insp_Start', 'Avg_Time_to_Next_Insp_Before_Laser']), inner='quartile')
    plt.title('Violin Plots of Time to Next Inspiration by Even Breath Phase Bins')
    plt.xlabel('Breath Phase Range')
    plt.xticks(ticks=range(len(bin_labels_pi_range)), labels=bin_labels_pi_range, rotation=45)
    plt.ylabel('Time to Next Inspiration (s)')
    plt.legend(title='Time Reference', labels=['After Laser', 'Before Laser'])
    plt.show()
