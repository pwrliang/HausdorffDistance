import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import re

from expr.draw.common import dataset_labels


def load_df(dir):
    cache_file = hashlib.md5(dir.encode()).hexdigest()
    # If cache exists, load from pickle
    if os.path.exists(cache_file) and False:
        df = pd.read_pickle(cache_file)
        print("Loaded DataFrame from cache.", cache_file)
    else:
        all_files = glob.glob(os.path.join(dir, '*.json'))
        # Otherwise, load from JSON files and serialize
        json_records = []
        for file in all_files:
            with open(file) as f:
                json_records.append(json.load(f))

        df = pd.json_normalize(json_records)

        # Serialize to pickle
        df.to_pickle(cache_file)
        print("Loaded DataFrame from JSON and saved to cache.")
    return df


def get_avg_time(df):
    time_columns = [col for col in df.columns if re.search(r'Running\.Repeat(\d+)\.ReportedTime', col)]
    return df[time_columns].mean(axis=1)


def draw_eb_effectiveness():
    variants = ("eb_gpu", "itk_cpu", "monai_cpu")
    variant_labels = ("EB", "ITK", "MONAI")

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4))

    dataset = "BraTS2020_ValidationData"
    ax.set_xlabel("Percentile (%)")
    ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    ax.set_title("Running Time Distribution")
    ax.set_yscale('log')
    linestyles = ['dotted', 'dashed', 'solid', ]
    for variant_idx in range(len(variants)):
        variant = variants[variant_idx]
        df = load_df(f"logs/run_all/{variant}/{dataset}")
        if variant == 'monai_cpu':
            y = get_avg_time(df)
        else:
            y = df['Running.AvgTime']
        y = y.values
        y.sort()
        percentiles = np.linspace(0, 100, len(y))
        ax.plot(percentiles, y, label=variant_labels[variant_idx], ls=linestyles[variant_idx], )
        mean = y.mean()  # or np.mean(arr)
        std = y.std(ddof=0)  # population std‑dev; use ddof=1 for sample
        median = np.median(y)
        print("Variant", variant)
        print(f"Mean   : {mean:.4f}")
        print(f"Std dev: {std:.4f}")
        print(f"Median : {median:.4f}")

        ax.legend(loc='upper left', handletextpad=0.3,
                  borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)
    fig.savefig("eb_time.pdf", format='pdf', bbox_inches='tight')
    plt.show()


draw_eb_effectiveness()
