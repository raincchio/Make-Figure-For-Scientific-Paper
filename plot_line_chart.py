import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Set Matplotlib font and style settings
plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'times',
    'font.size': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 10,
    # "axes.spines.right": False,
    # "axes.spines.top": False,
    # 'figure.figsize': (8, 3.5),
})

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

domain = 'domain1'
algos = ['algo1', 'algo2', 'algo3', 'algo4']
paths = ["task"]
domain_algo_plot, algo_domain_path, retrived_algos = get_parsed_dict(paths)

algos = [algo for algo in retrived_algos if algo not in algos]
for id, algo in enumerate(retrived_algos):
    if algo in algos:  # remove null results algo
        retrived_algos.pop(id)
algos.extend(retrived_algos)  # include more algo and keep the algo index not change

# COLORS = ['#77AC30', '#A56DB0', "#F0C04A", '#DE6C3A', '#2988C7', '#0000FF']
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']
MARKERS = ['*', 's', 'o', '^']
LINES = ['-', '--', ':']
metrics = ["metric1"]

# Plot line chart
plt.figure(figsize=(4.8, 4.2)) # width and height
# Plot lines
max_len = 0
max_y_value = 0
for algo in domain_algo_plot[domain]:
    data_path = algo_domain_path[algo + '-' + domain] # can  get multiple metrics result
    res_dict = get_results(data_path, metrics=metrics)
    single_metric_res = res_dict[metrics[0]]
    results = smooth_results(single_metric_res)
    mean = np.mean(results, axis=1)
    std = np.std(results, axis=1)

    x_vals = np.arange(len(mean))  # x axis item interval

    color = COLORS[algos.index(algo)]
    marker = MARKERS[algos.index(algo) % 4]
    line = LINES[algos.index(algo)//3]
    marker_num = 8
    makerevery = x_vals[-1] // marker_num
    plt.plot(x_vals, mean, label=algo, marker=marker, markevery=makerevery, markerfacecolor='none', markersize=5.5, markeredgewidth=1.5, color=color, linestyle=line)
    plt.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.3)

    max_len = max(max_len, x_vals[-1])
    max_y_value = max(max_y_value, np.max(results))

# Plot misc
plt.ylabel('reward')
plt.xlabel('steps')

x_ticks_interval = 1000  # just want three ticks, let it be max_len//3
x_tick_interval = 0.5
x_max_value = 3.1
x_ticks_values = np.arange(0, x_max_value, x_tick_interval)
x_ticks = [x * x_ticks_interval for x in x_ticks_values]
plt.xticks(x_ticks, x_ticks_values)
plt.xlim(0, x_ticks[-1])
plt.ylim(0, np.ceil(max_y_value))
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

plt.grid(True, linestyle='-', alpha=0.5)
lgd = plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=2, fancybox=False, framealpha=1, edgecolor='black',prop={'size': 8})
lgd.get_frame().set_alpha(None)
lgd.get_frame().set_facecolor((0, 0, 0, 0))
lgd.get_frame().set_linewidth(0.3)
plt.subplots_adjust(right=0.7)
plt.tight_layout(rect=(0, 0.1, 1, 1))

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
plt.text(x_ticks[-1], -0.1, 'M', ha='center', va='center', transform=ax.get_xaxis_transform())
timestr = time.strftime("%Y%m%d-%H%M%S")
plt.savefig('./result/'+timestr +'.pdf', bbox_inches='tight',  dpi=300)
# print('./plotting/pdf/'+task+'.pdf ','plot finished!')


