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


def parse_vary_dist(prefix):
    time = []

    for file in os.listdir(prefix):
        with open(os.path.join(prefix, file), "r") as f:
            dist = None
            cpu_hd = None
            cpu_par_hd = None
            gpu_hd = None
            rt_hd = None

            for line in f:
                m = re.search("CPU HausdorffDistance (.*?) Time: (.*?) ms", line)
                if m is not None:
                    dist = float(m.groups()[0])
                    cpu_hd = float(m.groups()[1])
                m = re.search("CPU Parallel HausdorffDistance .*? Time: (.*?) ms", line)
                if m is not None:
                    cpu_par_hd = float(m.groups()[0])
                m = re.search("GPU HausdorffDistance .*? Time: (.*?) ms", line)
                if m is not None:
                    gpu_hd = float(m.groups()[0])
                m = re.search("RT HausdorffDistance: .*? Time: (.*?) ms", line)
                if m is not None:
                    rt_hd = float(m.groups()[0])
            time.append({"dist": dist, "Serial": cpu_hd, "Parallel": cpu_par_hd, "GPU": gpu_hd, "RT": rt_hd})
    df = pd.DataFrame(time)
    df = df.sort_values(by="dist")
    df.set_index("dist", inplace=True)

    return df


def draw_vary_dist(prefix):
    df = parse_vary_dist(prefix)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.,))
    df.plot(kind="line", ax=ax)

    # for i, ax in enumerate(axes):
        # for j, line in enumerate(ax.get_lines()):
        #     line.set_marker(markers[j])
        #     line.set_color('black')

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


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0]) + "/../logs/vary_dist"

    draw_vary_dist(dir)
