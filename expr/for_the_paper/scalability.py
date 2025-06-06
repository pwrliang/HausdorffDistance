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

from matplotlib.pyplot import legend
from matplotlib.ticker import FuncFormatter


def load_df(dir):
    folder_path = Path(dir)
    json_files = list(folder_path.rglob("*.json"))

    # Otherwise, load from JSON files and serialize
    json_records = []
    for file in json_files:
        # if kw in str(file):
        with open(file) as f:
            json_records.append(json.load(f))

    df = pd.json_normalize(json_records)

    return df


# Define the formatting function
def millions_formatter(x, pos):
    if x >= 1e6:
        return f'{x * 1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x * 1e-3:.0f}K'
    else:
        return f'{x:.0f}'

def percent_formatter(x, pos):
    return f'{x * 100:.1f}%'


dists = ['uniform', 'gaussian']
dist_labels = ['Uniform', 'Gaussian']
variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
variant_labels = ("EB-GPU", "NN-RT", "X-HD")
plt.rcParams.update({'font.size': 15})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

linestyles = ['dotted', 'dashed', 'solid', ]

def draw_subfig(vary_type,  ax):
    dfs = []
    for i in range(len(variants)):

        df_data = pd.DataFrame()
        legend_pos = None
        if vary_type == "Count":
            df = load_df(f"logs/run_all/{variants[i]}/scal_vary_size")
            df_data[vary_type] = df['Input.Files'].apply(lambda x: x[0]['NumPoints'])
            legend_pos='upper left'
            ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        else:
            df = load_df(f"logs/run_all/{variants[i]}/scal_vary_translate")
            df_data[vary_type] = df['Input.Translate']
            legend_pos='upper right'
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        df_data[variant_labels[i]] = df['Running.AvgTime']
        df_data.set_index(vary_type, inplace=True)
        dfs.append(df_data)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=vary_type), dfs)
    merged_df.sort_values(vary_type, inplace=True)
    print(merged_df)
    merged_df.plot(ax=ax, style=[':', '--', '-'])
    ax.legend(loc=legend_pos, ncol=1, handletextpad=0.3,
              borderaxespad=0.2, frameon=False)

    ax.set_ylabel("Running Time (ms)")
    # ax.set_yscale('log')



draw_subfig('Count',  axes[0])
axes[0].set_title(f"Scalability - Varying # of Points")
axes[0].set_xlabel("Number of Points")
draw_subfig('Translate',  axes[1])
axes[1].set_title(f"Scalability - Varying Distance")
axes[1].set_xlabel("Translate x-axis to model size (%)")
fig.tight_layout(pad=0.1)
fig.savefig("scalability.pdf", format='pdf', bbox_inches='tight')
plt.show()
