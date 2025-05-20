import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import re


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


def enabled_hybrid(df):

    return df

def get_avg_time(df):
    time_columns = [col for col in df.columns if re.search(r'Running\.Repeat(\d+)\.ReportedTime', col)]
    return df[time_columns].mean(axis=1)


def draw_hybrid_vs_all():
    variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
    variant_labels = ("EB-GPU", "NN-RT", "Hybrid")

    datasets = ["BraTS2020_TrainingData", "ModelNet40"]
    dataset_labels = ["(a) BraTS", "(b) ModelNet"]
    linestyles = ['dotted', 'dashed', 'solid', ]

    plt.rcParams.update({'font.size': 15})
    # plt.rcParams['hatch.linewidth'] = 4
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(9, 4))

    for dataset_id in range(len(datasets)):
        dataset = datasets[dataset_id]
        ax = axes[dataset_id]
        ax.set_xlabel("Percentile (%)")
        ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
        ax.set_title(f"{dataset_labels[dataset_id]} dataset")
        ax.set_yscale('log')

        for variant_idx in range(len(variants)):
            variant = variants[variant_idx]
            df = load_df(f"logs/run_all/{variant}/{dataset}")
            # x = pd.Series([i for i in range(1, len(df) + 1)])
            y = get_avg_time(df)
            y = y.values
            y.sort()
            # if variant_idx == 2:
            #     print(y[:-1])
            percentiles = np.linspace(0, 100, len(y))
            ax.plot(percentiles, y, ls=linestyles[variant_idx], label=variant_labels[variant_idx], linewidth=2,)
            mean = y.mean()  # or np.mean(arr)
            std = y.std(ddof=0)  # population std‑dev; use ddof=1 for sample
            median = np.median(y)
            print("Variant", variant)
            print(f"Mean   : {mean:.4f}")
            print(f"Std dev: {std:.4f}")
            print(f"Median : {median:.4f}")
        # for i, line in enumerate(ax.get_lines()):
        #     line.set_marker(markers[i])
        ax.legend(loc='upper left', ncol=1, handletextpad=0.3,
                  borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)
    fig.savefig("hybrid_vs_all.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# draw_hybrid_vs_all()


def draw_hybrid_analysis():
    variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
    variant_labels = ("EB-GPU", "NN-RT", "Hybrid")
    variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
    variant_labels = ("EB-GPU", "NN-RT", "Hybrid")

    datasets = ["BraTS2020_TrainingData", "ModelNet40"]
    dataset_labels = ["(a) BraTS", "(b) ModelNet"]
    linestyles = ['dotted', 'dashed', 'solid', ]
    datasets = [ "ModelNet40"]
    dataset_labels = [ "(a) "]

    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams['hatch.linewidth'] = 4
    # fig, ax = plt.subplots(nrows=1, ncols=len(datasets), figsize=(9, 4))

    for dataset_id in range(len(datasets)):
        dataset = datasets[dataset_id]
        # ax = axes[dataset_id]
        # ax.set_xlabel("Percentile (%)")
        # ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
        # ax.set_title(f"{dataset_labels[dataset_id]} dataset")
        # ax.set_yscale('log')

        df_eb = load_df(f"logs/run_all/eb_gpu/{dataset}")
        df_rt = load_df(f"logs/run_all/rt_gpu/{dataset}")
        df_hybrid = load_df(f"logs/run_all/hybrid_gpu/{dataset}")

        df_hybrid = df_hybrid[df_hybrid["Running.Repeat0.Iter2.NumTermPoints"] > 0]
        file_names = df_hybrid[["Input.FileA.Path", "Input.FileB.Path"]]
        df_eb = pd.merge(df_eb, file_names, on=["Input.FileA.Path", "Input.FileB.Path"])
        df_rt = pd.merge(df_rt, file_names, on=["Input.FileA.Path", "Input.FileB.Path"])

        # x = pd.Series([i for i in range(1, len(df_eb) + 1)])
        s_eb = get_avg_time(df_eb)
        s_rt = get_avg_time(df_rt)
        s_hybrid = get_avg_time(df_hybrid).to_numpy()

        # ax.scatter(x, s_eb)
        # ax.scatter(x, s_rt)
        # ax.scatter(x, s_hybrid)


        s_best = np.minimum(s_eb, s_rt)
        speedups = (s_best / s_hybrid).to_numpy()
        print(speedups)
        best_idx = np.argmax(speedups)
        print(file_names.iloc[best_idx])
        print("best", s_best[best_idx], s_hybrid[best_idx])

        break

        # x = pd.Series([i for i in range(1, len(df) + 1)])
        # for variant_idx in range(len(variants)):
        #     variant = variants[variant_idx]
        #     df = load_df(f"logs/run_all/{variant}/{dataset}")
        #     # x = pd.Series([i for i in range(1, len(df) + 1)])
        #     y = get_avg_time(df)
        #     y = y.values
        #     y.sort()
            # if variant_idx == 2:
            #     print(y[:-1])
        #     percentiles = np.linspace(0, 100, len(y))
        #     ax.plot(percentiles, y, ls=linestyles[variant_idx], label=variant_labels[variant_idx], linewidth=2,)
        #     mean = y.mean()  # or np.mean(arr)
        #     std = y.std(ddof=0)  # population std‑dev; use ddof=1 for sample
        #     median = np.median(y)
        #     print("Variant", variant)
        #     print(f"Mean   : {mean:.4f}")
        #     print(f"Std dev: {std:.4f}")
        #     print(f"Median : {median:.4f}")
        # # for i, line in enumerate(ax.get_lines()):
        #     line.set_marker(markers[i])
    #     ax.legend(loc='upper left', ncol=1, handletextpad=0.3,
    #               borderaxespad=0.2, frameon=False)
    # fig.tight_layout(pad=0.1)
    # fig.savefig("hybrid_vs_all.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

draw_hybrid_analysis()
# draw_hybrid_vs_all()