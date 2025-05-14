import glob
import json

from matplotlib import pyplot as plt, image as mpimg
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import os


def draw_time():
    def load_df(files, cache_file):
        # If cache exists, load from pickle
        if os.path.exists(cache_file):
            df = pd.read_pickle(cache_file)
            print("Loaded DataFrame from cache.")
        else:
            # Otherwise, load from JSON files and serialize
            json_records = []
            for file in files:
                with open(file) as f:
                    json_records.append(json.load(f))

            df = pd.json_normalize(json_records)

            # Serialize to pickle
            df.to_pickle(cache_file)
            print("Loaded DataFrame from JSON and saved to cache.")
        return df

    all_files = glob.glob(os.path.join("logs/analysis", '*.json'))
    df = load_df(all_files, "intro.pkl")

    translate = 0.07

    df = df[(df['Input.Translate'] - translate) < 1e-5]

    df_eb = df[df["Running.Repeat0.Algorithm"] == "Early Break"]
    eb_histo = df_eb["Running.Repeat0.CmpHistogram"].iloc[0]

    df_rt = df[df["Running.Repeat0.Algorithm"] == "Ray Tracing"]
    rt_histo = df_rt["Running.Repeat0.Iter1.HitsHistogram"].iloc[0]

    x_eb = [entry['percentile'] for entry in eb_histo]
    y_eb = [entry['value'] for entry in eb_histo]

    x_rt = [entry['percentile'] for entry in rt_histo]
    y_rt = [entry['value'] for entry in rt_histo]

    plt.figure(figsize=(10, 6))
    plt.plot(x_eb, y_eb, marker='o')
    plt.plot(x_rt, y_rt, marker='o')
    plt.xlabel('Percentile')
    plt.ylabel('Value')
    plt.title('Value vs. Percentile')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # df_eb = df_eb[["Input.Translate", "Running.Repeat0.ComputeTime"]].sort_values(by="Input.Translate")
    # df_eb['Input.Translate'] = df_eb['Input.Translate'] * 100
    # df_eb.set_index('Input.Translate', inplace=True)
    # df_eb.rename(columns={'Running.Repeat0.ComputeTime': 'Early Break'}, inplace=True)
    #
    # df_nn = df[df["Running.Repeat0.Algorithm"] == "Nearest Neighbor Search"]
    # df_nn = df_nn[["Input.Translate", "Running.Repeat0.ComputeTime"]].sort_values(by="Input.Translate")
    # df_nn['Input.Translate'] = df_nn['Input.Translate'] * 100
    # df_nn.set_index('Input.Translate', inplace=True)
    # df_nn.rename(columns={'Running.Repeat0.ComputeTime': 'NN Search'}, inplace=True)

    # print(df_eb)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # img = mpimg.imread('rendered.png')  # or use PIL.Image.open if preferred
    # axes[0].imshow(img)
    # axes[0].axis('off')  # Hide axes
    # axes[0].set_title('(a) Two same chair models and its HD', y=-0.25)
    # axes[0].text(0.1, 0.08, "Hausdorff\nDistance", transform=axes[0].transAxes, fontsize=10, )
    #
    # df_eb.plot(kind='line', marker='o', ax=axes[1])
    # df_nn.plot(kind='line', marker='o', ax=axes[1])
    #
    # axes[1].margins(x=0.05, y=0.25)
    # axes[1].set_xlabel('Translate x-axis proportionally to model size (%)')
    # axes[1].set_ylabel('Running Time (ms)')
    # axes[1].set_title("(b) Running time by moving the blue model", y=-0.25)
    # axes[1].legend(loc='upper left', ncol=3, handletextpad=0.3,
    #                fontsize=11, borderaxespad=0.2, frameon=False)
    # plt.tight_layout()
    # fig.savefig('intro.pdf', format='pdf', bbox_inches='tight')
    # plt.show()


draw_time()
