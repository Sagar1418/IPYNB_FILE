import pandas as pd
import numpy as np
import re
from multiprocessing import Pool, cpu_count

station_master_file = '/media/sagarkumar/New Volume/SAGAR/IPYNB_FILE/DATA_GENERATION/ss_unique.csv'
switch_master_file = '/media/sagarkumar/New Volume/SAGAR/IPYNB_FILE/DATA_GENERATION/switch.csv'
output_file = '/media/sagarkumar/New Volume/SAGAR/IPYNB_FILE/DATA_GENERATION/HT_fault_cable_info_processed2.csv'

result_df = pd.read_csv(output_file)
station_master = pd.read_csv(station_master_file)
switch_master = pd.read_csv(switch_master_file)

all_station_names = station_master['SOURCE_SS'].dropna().str.upper().str.strip().tolist()
all_switch_numbers = switch_master['0'].dropna().astype(str).str.strip().tolist()

# Precompile regex patterns globally (so accessible inside worker function)
station_patterns = [(name, re.compile(r'\b' + re.escape(name) + r'\b')) for name in all_station_names]
switch_patterns = [(sw, re.compile(r'\b' + re.escape(sw) + r'\b')) for sw in all_switch_numbers]

def match_both(text):
    if pd.isnull(text):
        return np.nan, np.nan
    text_upper = str(text).upper()
    found_stations = [name for name, pattern in station_patterns if pattern.search(text_upper)]
    found_switches = [sw for sw, pattern in switch_patterns if pattern.search(text_upper)]
    # Return both as a tuple
    stations = '; '.join(found_stations) if found_stations else np.nan
    switches = '; '.join(found_switches) if found_switches else np.nan
    return stations, switches

def parallel_apply(series, func, n_cores=None):
    n_cores = n_cores or max(cpu_count() - 3, 1)
    with Pool(n_cores) as pool:
        # List of tuples (station, switch)
        result = pool.map(func, series.tolist())
    return result

if __name__ == "__main__":
    print("Using multiprocessing to extract AFFECTED_STATION and AFFECTED_SWITCH...")
    results = parallel_apply(result_df['REASON_TEXT'], match_both)
    result_df['AFFECTED_STATION'] = [r[0] for r in results]
    result_df['AFFECTED_SWITCH'] = [r[1] for r in results]
    result_df.to_csv(output_file, index=False)
    print("Updated AFFECTED_STATION and AFFECTED_SWITCH with multiprocessing and saved to:", output_file)
