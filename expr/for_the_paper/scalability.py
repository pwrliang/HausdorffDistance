import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import re
from pathlib import Path
from functools import reduce
from matplotlib.ticker import FuncFormatter

def load_df(dir, kw):
    folder_path = Path(dir)
    json_files = list(folder_path.rglob("*.json"))

    # Otherwise, load from JSON files and serialize
    json_records = []
    for file in json_files:
        if kw in str(file):
            with open(file) as f:
                json_records.append(json.load(f))

    df = pd.json_normalize(json_records)

    return df

# Define the formatting function
def millions_formatter(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    else:
        return f'{x:.0f}'

dists = ['uniform', 'gaussian']
dist_labels = ['Uniform', 'Gaussian']
variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
variant_labels = ("EB-GPU", "NN-RT", "Hybrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))


def draw_subfig(dist, dist_labels, ax):
    dfs = []
    for i in range(len(variants)):
        df = load_df(f"logs/run_all/{variants[i]}/scalability", dist)
        df_data = pd.DataFrame()
        df_data['Count'] = df['Input.Files'].apply(lambda x:x[0]['NumPoints'])
        # df_data['Dataset'] = df['Input.Files'].apply(
        #     lambda x: os.path.basename(x[0]['Path']) + '-' +
        #               os.path.basename(x[1]['Path']))
        df_data[variant_labels[i]] = df['Running.AvgTime']
        df_data.set_index('Count', inplace=True)
        dfs.append(df_data)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Count'), dfs)
    merged_df.sort_values('Count', inplace=True)
    print(merged_df)
    merged_df.plot( ax=ax, )
    ax.set_xlabel("Number of Points")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

draw_subfig('uniform', dist_labels, axes[0])
draw_subfig('gaussian', dist_labels, axes[1])
plt.show()