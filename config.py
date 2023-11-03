# config.py serves as a place to store variables that will be shared in multiple scripts
__all__ = ['FOLDER_PATH', 'FILE_PATH', 'EVENT_DURATION_MS', 'NUM_PHASE_BINS', 'NUM_BREATHS_BEFORE', 'SAMPLE_RATE']

# Common file paths
FOLDER_PATH = r"data\nk1roprm1-stgtacr-co1-wav"
FILE_PATH = r"C:\Users\Beatriz\data\nk1roprm1-stgtacr-co1-wav\588_10_nk1roprm1_stgtacr2_11012023_10ms_10mW.WAV"

# Laser stim parameters
EVENT_DURATION_MS = 10 # Defaults to 1ms unless specified

#Plotting Parameters
NUM_PHASE_BINS = 20 # How many bins do you want ti split 0-2pi into?
NUM_BREATHS_BEFORE = 2 # How many breaths do you want to be averaged for the before laser baseline?

# Data collection parameters
SAMPLE_RATE = 1000 # Where (1000 = 1k/s)