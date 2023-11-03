# Plot accurate breath phases for two example laser events using the previous function outputs
first_two_laser_event_starts = [event[0] for event in filtered_laser_events[:2]]
for i, laser_event_start in enumerate(first_two_laser_event_starts, start=1):
    plot_breath_phases(laser_event_start, full_breath_cycles, expiration_indices, filtered_breathing_signal, title_suffix=f"#{i}")



example_laser_event_start = filtered_laser_events[0][0]
example_time_to_next_insp, example_laser_breath_insp_start, example_next_insp_idx = time_to_next_inspiration_from_insp_start(example_laser_event_start, inspiration_indices, sample_rate)

# Correct the plot to extend the red line from the start of the breath in which the laser event occurred to the start of the next inspiration
plt.figure(figsize=(12, 6))
plt.plot(np.arange(example_laser_breath_insp_start, example_next_insp_idx), filtered_breathing_signal[example_laser_breath_insp_start:example_next_insp_idx], label="Filtered Breathing Signal")
plt.axvline(x=example_laser_event_start, color='m', linestyle='--', label=f"Laser Event Start")
plt.axvline(x=example_next_insp_idx, color='r', linestyle='--', label=f"Next Inspiration Start")

# Draw a red line extending horizontally to indicate the time to the next inspiration, now starting from the beginning of the breath
plt.hlines(y=filtered_breathing_signal[example_laser_breath_insp_start], xmin=example_laser_breath_insp_start, xmax=example_next_insp_idx, colors='r', linestyles='-', label="Time to Next Inspiration")

plt.title(f"Corrected: Example Laser Event and Time to Next Inspiration")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
