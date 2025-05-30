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


def load_df(dir):
    folder_path = Path(dir)
    json_files = list(folder_path.rglob("*.json"))

    # Otherwise, load from JSON files and serialize
    json_records = []
    for file in json_files:
        with open(file) as f:
            json_records.append(json.load(f))

    df = pd.json_normalize(json_records)

    return df


def enabled_hybrid(df):
    return df


def get_avg_time(df):
    time_columns = [col for col in df.columns if re.search(r'Running\.Repeat(\d+)\.ReportedTime', col)]
    return df[time_columns].mean(axis=1)


def draw_mri_modelnet():
    variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
    variant_labels = ("EB-GPU", "NN-RT", "Hybrid")

    datasets = ["BraTS2020_ValidationData", "ModelNet40"]
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

        print("dataset", dataset, '\n')

        df_rt = None
        df_hybrid = None

        for variant_idx in range(len(variants)):
            variant = variants[variant_idx]
            df = load_df(f"logs/run_all/{variant}/{dataset}")
            if variant_idx == 1:
                df_rt = df
            elif variant_idx == 2:
                df_hybrid = df
            # x = pd.Series([i for i in range(1, len(df) + 1)])
            y = df["Running.AvgTime"]
            y = y.values
            y.sort()

            percentiles = np.linspace(0, 100, len(y))
            ax.plot(percentiles, y, ls=linestyles[variant_idx], label=variant_labels[variant_idx], linewidth=2, )

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
        slow_rows = df_hybrid[df_hybrid['Running.AvgTime'] > df_rt['Running.AvgTime']]
        # print(slow_rows)

    fig.tight_layout(pad=0.1)
    fig.savefig("hybrid_vs_all.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def draw_spatial_graphics():
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams['hatch.linewidth'] = 4
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    variants = ("eb_gpu", "rt_gpu", "hybrid_gpu")
    variant_labels = ("EB-GPU", "NN-RT", "Hybrid")
    geo_dataset_labels = {'USADetailedWaterBodies.wkt': 'USWater', 'USACensusBlockGroupBoundaries.wkt': 'USBlock',
                          'lakes.bz2.wkt': 'OSMLakes', 'parks.bz2.wkt': 'OSMParks', 'dtl_cnty.wkt': 'USCounty',
                          'uszipcode.wkt': 'USZipcode'}

    graphics_dataset_labels = {'dragon.ply':'Dragon','asian_dragon.ply':'Asian Dragon',
                               'thai_statuette.ply': 'Thai','happy_buddha.ply':'Buddha',}

    def draw_subfig(dataset_name, dataset_labels, ax):
        dfs = []
        for i in range(len(variants)):
            df = load_df(f"logs/run_all/{variants[i]}/{dataset_name}")
            df_data = pd.DataFrame()
            df_data['Dataset'] = df['Input.Files'].apply(
                lambda x: dataset_labels[os.path.basename(x[0]['Path'])] + '-' + dataset_labels[
                    os.path.basename(x[1]['Path'])])
            df_data[variant_labels[i]] = df['Running.AvgTime']
            df_data.set_index('Dataset', inplace=True)
            dfs.append(df_data)
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='Dataset'), dfs)
        merged_df.plot(kind='bar', ax=ax, )

    draw_subfig("geo", geo_dataset_labels, axes[0])
    draw_subfig("graphics", graphics_dataset_labels, axes[1])

    fig.tight_layout(pad=0.1)
    fig.savefig("hybrid_vs_all.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# draw_hybrid_vs_all()


def draw_hybrid_analysis():
    df = load_df("logs/run_all/effectiveness")

    run_eb = None
    run_rt = None
    run_hybrid = None

    for repeats in df["Running.Repeats"]:
        repeat = repeats[0]
        algo = repeat["Algorithm"]
        if algo == "Hybrid":
            run_hybrid = repeat
        elif algo == "Ray Tracing":
            run_rt = repeat
        elif algo == "Early Break":
            run_eb = repeat

    iterations_rt = run_rt["Iterations"]
    iterations_hybrid = run_hybrid["Iterations"]

    iterations = [i for i in range(0, len(iterations_rt), 1)]
    rt_time_per_iter = [iterations_rt[i]['RTTime'] for i in iterations]
    hybrid_time_per_iter = [iterations_hybrid[i]['RTTime'] for i in iterations]
    iterations = [x + 1 for x in iterations]

    def get_draw_data(iterations):
        iter_nums = []
        wl_in = []
        wl_out = []
        wl_term = []
        rt_time = []
        for iteration in iterations:
            iter_nums.append(iteration["Iteration"])
            wl_in.append(iteration["NumInputPoints"])
            wl_out.append(iteration["NumOutputPoints"])
            rt_time.append(iteration["RTTime"])
            if "NumTermPoints" in iteration:
                wl_term.append(iteration["NumTermPoints"])
        data = {'Iteration': iter_nums,
                'Time': rt_time}
        # if len(wl_term) > 0:
        #     data['Terminated'] = wl_term
        df = pd.DataFrame(data)
        df.set_index('Iteration', inplace=True)
        return df

    df_hybrid = pd.DataFrame({'Iteration': iterations, 'NN-RT': rt_time_per_iter, 'EarlyTerm': hybrid_time_per_iter})
    df_hybrid.set_index('Iteration', inplace=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.,), gridspec_kw={'width_ratios': [6, 6]})

    df_hybrid.plot(kind='line', ax=axes[0], alpha=0.7, linewidth=2, color=['orange', 'green'])
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel('Time (ms)', fontsize=14)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(left=0)
    axes[0].set_title("(a) Running time of NN-RT and Hybrid\nin each iteration", fontsize=16)
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)

    breakdown = {'Methods': ['EB-GPU', 'NN-RT', 'Hybrid'],
                 'GridConst.': [0, run_rt['Grid']['BuildTime'], run_hybrid['Grid']['BuildTime']],
                 'BVHConst.': [0, run_rt['BVHBuildTime'], run_hybrid['BVHBuildTime']],
                 'EB-GPU': [run_eb['ReportedTime'], 0, run_hybrid['EBTime']],
                 'NN-RT': [0, sum([iterations_rt[i]['RTTime'] for i in range(0, len(iterations_rt))]),
                           sum([iterations_hybrid[i]['RTTime'] for i in range(0, len(iterations_hybrid))]), ],
                 }
    df_breakdown = pd.DataFrame(breakdown)
    df_breakdown.set_index('Methods', inplace=True)
    df_breakdown.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('(b) Running Time Breakdown', fontsize=16)
    axes[1].set_ylabel('Time (ms)', fontsize=14)
    axes[1].set_xlabel("Method", fontsize=14)
    axes[1].set_xticklabels(df_breakdown.index, rotation=0)
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)

    for label in axes[1].get_xticklabels():
        label.set_fontweight('bold')

    # Add labels and title for clarity
    # plt.title('Stacked Bar Chart Example')

    # plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.legend()
    plt.tight_layout(pad=0.5)  # Adjust layout to prevent labels from overlapping
    plt.show()

    # print("run_hybrid", run_hybrid)
    # print("run_rt", run_rt)
    # df_eb = df[df["Running.Repeats.Algorithm"] == "Early Break"].filter(like='Repeat0')
    # df_hybrid = df[df["Running.Repeat0.Algorithm"] == "Hybrid"]
    # df_rt = df[df["Running.Repeat0.Algorithm"] == "Ray Tracing"].filter(like='Repeat0')

    # df_rt_iters = df_rt.filter(like="Iter").T

    # print(df_rt_iters)

    # print(df)
    # datasets = ["BraTS2020_TrainingData", "ModelNet40"]
    # dataset_labels = ["(a) BraTS", "(b) ModelNet"]
    # linestyles = ['dotted', 'dashed', 'solid', ]
    # datasets = [ "BraTS2020_TrainingData"]
    # dataset_labels = [ "(a) BraTS"]

    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams['hatch.linewidth'] = 4
    # fig, ax = plt.subplots(nrows=1, ncols=len(datasets), figsize=(9, 4))

    # for dataset_id in range(len(datasets)):
    #     dataset = datasets[dataset_id]
    #     # ax = axes[dataset_id]
    #     # ax.set_xlabel("Percentile (%)")
    #     # ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    #     # ax.set_title(f"{dataset_labels[dataset_id]} dataset")
    #     # ax.set_yscale('log')
    #
    #     df_eb = load_df(f"logs/run_all/eb_gpu/{dataset}")
    #     df_rt = load_df(f"logs/run_all/rt_gpu/{dataset}")
    #     df_hybrid = load_df(f"logs/run_all/hybrid_gpu/{dataset}")
    #
    #     df_hybrid = df_hybrid[df_hybrid["Running.Repeat0.Iter2.NumTermPoints"] > 0]
    #     file_names = df_hybrid[["Input.FileA.Path", "Input.FileB.Path"]]
    #     df_eb = pd.merge(df_eb, file_names, on=["Input.FileA.Path", "Input.FileB.Path"])
    #     df_rt = pd.merge(df_rt, file_names, on=["Input.FileA.Path", "Input.FileB.Path"])
    #
    #     # x = pd.Series([i for i in range(1, len(df_eb) + 1)])
    #     s_eb = get_avg_time(df_eb)
    #     s_rt = get_avg_time(df_rt)
    #     s_hybrid = get_avg_time(df_hybrid).to_numpy()
    #
    #     # ax.scatter(x, s_eb)
    #     # ax.scatter(x, s_rt)
    #     # ax.scatter(x, s_hybrid)
    #
    #
    #     s_best = np.minimum(s_eb, s_rt)
    #     speedups = (s_best / s_hybrid).to_numpy()
    #     print(speedups)
    #     best_idx = np.argmax(speedups)
    #     print(file_names.iloc[best_idx])
    #     print("best", s_best[best_idx], s_hybrid[best_idx])
    #
    #     break

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


# draw_hybrid_analysis()
draw_mri_modelnet()
draw_spatial_graphics()
