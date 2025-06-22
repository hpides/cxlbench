import glob
import json
import os
import pandas as pd

def load_benchmarks_from_json_files(result_path):
    all_benchmarks = []
    
    if os.path.isdir(result_path):
        json_files = glob.glob(os.path.join(result_path, '*.json'))
    else:
        json_files = [result_path]
    
    for file_path in json_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if 'benchmarks' in data:
                all_benchmarks.extend(data['benchmarks'])

    benchmarks_df = pd.DataFrame(all_benchmarks)
    
    return benchmarks_df

def load_all_csv(result_path):
    data_frames = []
    
    if os.path.isdir(result_path):
        files = glob.glob(os.path.join(result_path, '*.csv'))
    else:
        files = [result_path]
    
    for file_path in files:
        id = file_path.rsplit("/",1)[-1][:-4]
        with open(file_path, 'r') as file:
            data = pd.read_csv(file_path, on_bad_lines='skip')
            data["id"] = id
            data_frames.append(data)
    result_df = pd.concat(data_frames)
    return result_df
