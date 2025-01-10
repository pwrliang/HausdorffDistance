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


def scale_size(size_list, k_scale=1000):
    return tuple(str(int(kb)) + "K" if kb < k_scale else str(int(kb / k_scale)) + "M" for kb in
                 np.asarray(size_list) / k_scale)


def parse_vary_dist(prefix, variant):
    time = []

    for file in os.listdir(prefix):
        if file.startswith(variant):
            with open(os.path.join(prefix, file), "r") as f:
                dist = None
                running_time = None
                visited_pairs = None

                for line in f:
                    m = re.search("Running Time (.*?) ms", line)
                    if m is not None:
                        running_time = float(m.groups()[0])
                    m = re.search("HausdorffDistance: distance is (.*?)$", line)
                    if m is not None:
                        dist = float(m.groups()[0])
                    m = re.search("Answer: (.*?) Diff:", line)
                    if m is not None:
                        dist = float(m.groups()[0])
                    m = re.search("Visited Point Pairs (.*?) ", line)
                    if m is not None:
                        visited_pairs = int(m.groups()[0])
                    m = re.search("Compared Pairs: (.*?)$", line)
                    if m is not None:
                        visited_pairs = int(m.groups()[0])
                time.append({"dist": dist, "Time": running_time, "Visited Pairs": visited_pairs})
    df = pd.DataFrame(time)
    df = df.sort_values(by="dist")
    df.set_index("dist", inplace=True)

    return df


def draw_vary_dist(prefix):
    df_gpu = parse_vary_dist(prefix, "gpu")
    df_serial = parse_vary_dist(prefix, "serial")
    df_parallel = parse_vary_dist(prefix, "parallel")
    df_lbvh = parse_vary_dist(prefix, "lbvh")
    df_rt = parse_vary_dist(prefix, "rt")

    df = pd.concat([df_gpu, df_serial['Time'], df_parallel['Time'],
                    df_lbvh['Time'], df_rt['Time']], axis=1)

    df.columns = ["GPU", "Serial", "Parallel", "LBVH", "RT"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.,))
    df.plot(kind="line", ax=ax)

    # for i, ax in enumerate(axes):
    #     for j, line in enumerate(ax.get_lines()):
    #         line.set_marker(markers[j])
    #         line.set_color('black')

    ax.set_xlabel("Hausdorff Distance")
    ax.set_ylabel(ylabel='Query Time (ms)', labelpad=1)
    ax.set_yscale('log')
    ax.margins(x=0.05, y=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncol=4, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)

    fig.savefig("vary_dist.png", format='png', bbox_inches='tight')
    plt.show()
def draw_visited_pairs(prefix):
    df_serial = parse_vary_dist(prefix, "serial_dtl_cnty.wkt_dtl_cnty.wkt_")
    df_lbvh = parse_vary_dist(prefix, "lbvh_dtl_cnty.wkt_dtl_cnty.wkt_")

    df = pd.concat([df_serial, df_lbvh['Visited Pairs']], axis=1)
    df.drop('Time', axis=1, inplace=True)

    df.columns = ["Serial", "LBVH",]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.,))
    df.plot(kind="line", ax=ax)

    # for i, ax in enumerate(axes):
    #     for j, line in enumerate(ax.get_lines()):
    #         line.set_marker(markers[j])
    #         line.set_color('black')

    ax.set_xlabel("Hausdorff Distance")
    ax.set_ylabel(ylabel='Visited Pairs', labelpad=1)
    ax.set_yscale('log')
    ax.margins(x=0.05, y=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncol=4, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)

    fig.savefig("visited_pairs.png", format='png', bbox_inches='tight')
    plt.show()


def draw_running_time(prefix):
    df_serial = parse_vary_dist(prefix, "serial_dtl_cnty.wkt_dtl_cnty.wkt_")
    df_lbvh = parse_vary_dist(prefix, "lbvh_dtl_cnty.wkt_dtl_cnty.wkt_")

    df = pd.concat([df_serial, df_lbvh['Time']], axis=1)
    df.drop('Visited Pairs', axis=1, inplace=True)

    df.columns = ["Serial", "LBVH",]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.,))
    df.plot(kind="line", ax=ax)

    # for i, ax in enumerate(axes):
    #     for j, line in enumerate(ax.get_lines()):
    #         line.set_marker(markers[j])
    #         line.set_color('black')

    ax.set_xlabel("Hausdorff Distance")
    ax.set_ylabel(ylabel='Running Time (ms)', labelpad=1)
    ax.set_yscale('log')
    ax.margins(x=0.05, y=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncol=4, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    fig.tight_layout(pad=0.1)

    fig.savefig("running_time.png", format='png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0]) + "/../logs/vary_dist"

    # draw_vary_dist(dir)
    # draw_visited_pairs(dir)
    draw_running_time(dir)