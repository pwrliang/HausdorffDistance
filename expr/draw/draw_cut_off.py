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


def parse_histo(prefix, variant, pattern):
    histos = {}
    for file in os.listdir(prefix):
        if file.startswith(variant):

            m = re.search(pattern, file)
            if m is not None:
                df = pd.read_csv(os.path.join(prefix, file))
                histos[file] = df

    return histos


def draw_hit_histo(prefix, pattern):
    df_rt_gpu = parse_histo(prefix, "rt_gpu", pattern)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    i = 0
    cut_off_x = 128
    percentile_at_cut_off = []
    for file, df in df_rt_gpu.items():
        plt.plot(df["Value"], df["Percentile"])
        percentile_y = np.interp(cut_off_x, df["Value"], df["Percentile"])
        percentile_at_cut_off.append(percentile_y)
        i += 1
        # if i >= 20:
        #     break
    # Draw a vertical line at x = 3
    plt.axvline(x=cut_off_x, color='red', linestyle='--', linewidth=2, label="x = 100")
    mean_percentile = np.mean(percentile_at_cut_off)
    median_percentile = np.median(percentile_at_cut_off)
    print("Total pairs ", len(percentile_at_cut_off))
    print("Cut-off = {} Mean percentile = {}, Median percentile = {}".format(cut_off_x, mean_percentile,
                                                                             median_percentile))
    ax.set_xlabel("Number of Hits")
    ax.set_ylabel(ylabel='Percentile of points', labelpad=1)
    ax.set_xscale('log')
    fig.tight_layout(pad=0.1)

    fig.savefig("hits_percentile.png", format='png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0]) + "/../logs/BraTS20"

    # draw_hit_histo(dir, 'BraTS20_Training_(.*?)_t1.nii_BraTS20_Training_(.*?)_t1.nii_iter_1.log')
    dir = os.path.dirname(sys.argv[0]) + "/../logs/vary_datasets"
    draw_hit_histo(dir, 'rt_gpu_(.*?).wkt_(.*?).wkt_limit_500000_iter_1.log')
