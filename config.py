from pathlib import Path

# Source directory
src = Path.cwd()
print(f"Source path (cwd): {src}")

# Constants
MIN_EVENTS = 2
MAX_EVENTS = 30
SEED = 42           # SEED for replicability

# Data Directories
dir_data = src / 'data'
dir_data_ext = dir_data / 'external'
dir_data_int = dir_data / 'interim'
dir_data_pro = dir_data / 'processed'
dir_data_raw = dir_data / 'raw'

# Model Directories
dir_models = src.parent / 'models'

# Other Directories
dir_results = src.parent / 'results'

# Log Directories
dir_logs = src / 'logs'
dir_hparams = dir_logs / 'hparams'
dir_hparam_tuning = dir_logs / 'hparams_tuning'

# enable/disable console output
verbose = True
if verbose:
    print("Verbose mode is activated!")

# enable/disable tests
test_mode = True
if test_mode:
    print("Tests mode is activated!")
