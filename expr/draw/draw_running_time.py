import matplotlib
import matplotlib.pyplot as plt
import math
import os
import numpy as np
# import comm_settings
from common import datasets, dataset_labels, hatches, markers, linestyles
import sys
import re
import pandas as pd
import json


def scale_size(size_list, k_scale=1000):
    return tuple(str(int(kb)) + "K" if kb < k_scale else str(int(kb / k_scale)) + "M" for kb in
                 np.asarray(size_list) / k_scale)


def parse_vary_dist(prefix, variant, pattern):
    time = []

    for file in os.listdir(prefix):
        m = re.search(variant + "_" + pattern, file)
        if m is not None:
            file_id1 = m.groups()[0].strip()
            file_id2 = m.groups()[1].strip()
            file_id = file_id1 + "-" + file_id2

            with open(os.path.join(prefix, file), "r") as f:
                dist = None
                running_time = None
                voxels = None

                for line in f:
                    m = re.search("Running Time (.*?) ms$", line)
                    if m is not None:
                        running_time = float(m.groups()[0])
                    m = re.search("HausdorffDistance: distance is (.*?)$", line)
                    if m is not None:
                        dist = float(m.groups()[0])
                    m = re.search("Valid Voxels (.*?)$", line)
                    if m is not None:
                        voxels = float(m.groups()[0])
                time.append({'file_id': file_id, "dist": dist, "time": running_time, "voxels": voxels})
    df = pd.DataFrame(time)
    df = df.sort_values(by="file_id")
    df.set_index("file_id", inplace=True)

    return df


def parse_histo(prefix, variant):
    histos = {}
    for file in os.listdir(prefix):
        if file.startswith(variant):
            file_id = None

            m = re.search('BraTS20_Training_(.*?)_t1.nii_BraTS20_Training_(.*?)_t1.nii_iter_1.log', file)
            if m is not None:
                file_id1 = m.groups()[0]
                file_id2 = m.groups()[1]
                file_id = file_id1 + "-" + file_id2
                df = pd.read_csv(os.path.join(prefix, file))
                histos[file] = df

    return histos


def parse_json_logs(prefix, variant):
    cache_file = variant + ".pkl"
    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
        print(f"Loaded DataFrame from {cache_file}.")
    else:
        dir = os.path.join(prefix, variant)
        json_records = []
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            with open(path) as f:
                json_records.append(json.load(f))
        df = pd.json_normalize(json_records)
        df.to_pickle(cache_file)
        print(f"Loaded DataFrame from JSON and saved to {cache_file}.")
    return df


def draw_running_time_box(prefix):
    df_itk = parse_vary_dist(prefix, "itk")
    df_eb_parallel = parse_vary_dist(prefix, "eb_parallel")
    df_eb_gpu = parse_vary_dist(prefix, "eb_gpu")
    df_zorder_gpu = parse_vary_dist(prefix, "zorder_gpu")
    df_rt_gpu = parse_vary_dist(prefix, "rt_gpu")
    df_hybrid_gpu = parse_vary_dist(prefix, "hybrid_gpu")

    labels = ['ITK', 'EB-Parallel', 'EB-GPU', 'Zorder-GPU', 'RT-GPU', 'Hybrid-GPU']
    dfs = [df_itk, df_eb_parallel, df_eb_gpu, df_zorder_gpu, df_rt_gpu, df_hybrid_gpu]

    for i in range(len(dfs)):
        label = labels[i]
        df = dfs[i]
        df.rename(columns={"time": label}, inplace=True)
        df.drop(['dist', 'voxels'], axis=1, inplace=True)
    wide_df = pd.concat(dfs, axis=1)
    print(wide_df)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    # Box plot
    wide_df.boxplot(grid=False, ax=ax)

    # cax = ax.matshow(wide_df, cmap='coolwarm', aspect='auto')

    # Add colorbar
    # wide_df = wide_df.head(20)
    # plt.colorbar(cax, label="Running Time (s)")
    # ax.set_xticks(range(wide_df.shape[1]))
    # ax.set_xticklabels(wide_df.columns, rotation=45)
    # ax.set_yticks([])  # Hides dataset labels for cleaner display

    # for i, ax in enumerate(axes):
    #     for j, line in enumerate(ax.get_lines()):
    #         line.set_marker(markers[j])
    #         line.set_color('black')

    # ax.set_xlabel("Methods")
    # ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    # ax.set_yscale('log')
    # ax.margins(x=0.05, y=0.25)
    # ax.set_ylim(bottom=0)
    # ax.legend(loc='upper left', ncol=3, handletextpad=0.3,
    #           fontsize=11, borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)

    fig.savefig("running_time.png", format='png', bbox_inches='tight')
    plt.show()


def draw_running_time_dot(prefix):
    limit = 10000
    df_hybrid = parse_json_logs(prefix, "hybrid").head(limit)
    df_hybrid_auto = parse_json_logs(prefix, "hybrid_autotune").head(limit)
    df_eb = parse_json_logs(prefix, "eb").head(limit)
    df_itk = parse_json_logs(prefix, "itk").head(limit)
    labels = ['Hybrid', 'HybridAuto', 'EB', 'ITK']
    dfs = [df_hybrid, df_hybrid_auto, df_eb, df_itk]
    time_column = "Running.AvgTime"

    mean_times = []
    median_times = []

    for i in range(len(dfs)):
        label = labels[i]
        df = dfs[i]
        mean_times.append(np.mean(df[time_column]))
        median_times.append(np.median(df[time_column]))
        # df.rename(columns={"time": label}, inplace=True)
        # df.drop(['dist', 'voxels'], axis=1, inplace=True)
    print("Mean", mean_times)
    print("Median", median_times)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    # x_pos = np.arange(len(labels))
    # bar_width = 0.35
    # ax.bar(x_pos - bar_width / 2, mean_times, width=bar_width, label='Mean Time')
    # ax.bar(x_pos + bar_width / 2, median_times, width=bar_width, label='Median Time')
    # ax.set_xticks(x_pos, labels)
    # ax.set_xlabel("Variants")
    # ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    # auto_is_faster_count = (df_hybrid[time_column] > df_hybrid_auto[time_column]).sum()
    # num_rows = df_hybrid.shape[0]
    # print("Faster percent", auto_is_faster_count / num_rows * 100)

    df_itk[time_column].plot(marker='o', markersize=2, linestyle='', ax=ax, label="ITK")
    df_eb[time_column].plot(marker='o', markersize=2, linestyle='', ax=ax, label="EB")
    # df_hybrid[time_column].plot(marker='o', markersize=3, linestyle='', ax=ax, label="Hybrid")
    df_hybrid_auto[time_column].plot(marker='o', markersize=3, linestyle='', ax=ax, label="Hybrid")



    ax.set_yscale("log")
    ax.set_xlabel("Datasets")
    ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    ax.set_xticklabels([])  # Remove x tick labels

    ax.legend()
    fig.tight_layout(pad=0.1)

    # fig.savefig("running_time_all.png", format='png', bbox_inches='tight')
    plt.show()


def draw_running_time_by_dist(prefix):
    df_itk = parse_vary_dist(prefix, "itk")
    df_eb_parallel = parse_vary_dist(prefix, "eb_parallel")
    df_eb_gpu = parse_vary_dist(prefix, "eb_gpu")
    df_zorder_gpu = parse_vary_dist(prefix, "zorder_gpu")
    df_rt_gpu = parse_vary_dist(prefix, "rt_gpu")
    df_hybrid_gpu = parse_vary_dist(prefix, "hybrid_gpu")

    labels = ['ITK', 'EB-Parallel', 'EB-GPU', 'Zorder-GPU', 'RT-GPU', 'Hybrid-GPU']
    dfs = [df_itk, df_eb_parallel, df_eb_gpu, df_zorder_gpu, df_rt_gpu, df_hybrid_gpu]

    for i in range(len(dfs)):
        label = labels[i]
        df = dfs[i]
        df.rename(columns={"time": label}, inplace=True)
        if i != 0:
            df.drop(['dist', 'voxels'], axis=1, inplace=True)
    wide_df = pd.concat(dfs, axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    wide_df = wide_df.sort_values(by='dist').drop('voxels', axis=1).set_index("dist")
    wide_df.plot(kind="line", ax=ax)
    # Box plot
    # wide_df.boxplot(grid=False, ax=ax)

    # cax = ax.matshow(wide_df, cmap='coolwarm', aspect='auto')

    # Add colorbar
    # wide_df = wide_df.head(20)
    # plt.colorbar(cax, label="Running Time (s)")
    # ax.set_xticks(range(wide_df.shape[1]))
    # ax.set_xticklabels(wide_df.columns, rotation=45)
    # ax.set_yticks([])  # Hides dataset labels for cleaner display

    # for i, ax in enumerate(axes):
    #     for j, line in enumerate(ax.get_lines()):
    #         line.set_marker(markers[j])
    #         line.set_color('black')

    # ax.set_xlabel("Methods")
    # ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    ax.set_yscale('log')
    # ax.margins(x=0.05, y=0.25)
    # ax.set_ylim(bottom=0)
    # ax.legend(loc='upper left', ncol=3, handletextpad=0.3,
    #           fontsize=11, borderaxespad=0.2, frameon=False)
    # fig.tight_layout(pad=0.1)
    # fig.savefig("running_time.png", format='png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # dir = os.path.dirname(sys.argv[0]) + "/../logs/BraTS20"
    # draw_running_time_dot(dir, 'BraTS20_Training_(.*?)_t1.nii_BraTS20_Training_(.*?)_t1.nii')
    # dir = os.path.dirname(sys.argv[0]) + "/../logs/vary_datasets"
    draw_running_time_dot('/Users/liang/BraTS2020_ValidationData')
