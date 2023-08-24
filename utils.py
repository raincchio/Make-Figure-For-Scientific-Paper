import os
import random
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib font and style settings
rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'ytick.right': True,
    'xtick.top': True,
    "font.family": "times",
    "font.size": 10,
    'axes.titlesize': 10,
    "legend.fontsize": 10,
    # "axes.spines.right": False,
    # "axes.spines.top": False,
    # 'figure.figsize': (8, 3.5),
}
plt.rcParams.update(rc_fonts)
plt.rc('axes', unicode_minus=False)

def get_results(algo_domain_path, metrics):
    res_dict = {}
    seeds = os.listdir(algo_domain_path)
    for seed in seeds:
        csv_path = os.path.join(algo_domain_path, seed, 'progress.csv')
        print('load file:', csv_path)
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=float)
        for metric in metrics:
            if metric in res_dict.keys():
                res_dict[metric].append(data[metric])
            else:
                res_dict[metric] = [data[metric]]
    for metric in metrics:
        min_row = min([len(col) for col in res_dict[metric]])
        clip_res = [col[0:min_row] for col in res_dict[metric]]
        res_dict[metric] = np.stack(clip_res, -1)
    return res_dict

def smooth_results(results, smoothing_window=100):
    smoothed = np.zeros_like(results)
    for idx in range(len(smoothed)):
        if idx == 0:
            smoothed[idx] = results[idx]
            continue
        start_idx = max(0, idx - smoothing_window)
        smoothed[idx] = np.mean(results[start_idx:idx], axis=0)
    return smoothed

def get_parsed_dict(paths):
    """
    path format: .../task/algo/domain/seed/progress.csv
    input: paths is a list of mlutiple task paths
    """
    domain_algo_plot = {}  # dict for plotting figure
    algo_domain_path = {}  # dict for loading data
    algos = []             # list for collecting all algo
    for path in paths:
        algos_ = os.listdir(path)
        algos.extend(algos_)
        for algo in algos_:
            domains = os.listdir(os.path.join(path, algo))
            for domain in domains:
                if domain not in domain_algo_plot.keys():
                    domain_algo_plot[domain] = [algo]
                else:
                    domain_algo_plot[domain].append(algo)
                assert algo + '-' + domain not in algo_domain_path.keys()
                algo_domain_path[algo + '-' + domain] = os.path.join(path, algo, domain)
    print('retrieved algos', algos)
    return domain_algo_plot, algo_domain_path, algos