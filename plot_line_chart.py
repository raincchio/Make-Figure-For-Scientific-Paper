import matplotlib.pyplot as plt
import numpy as np
from utils import get_results, get_parsed_dict, smooth_results


domains = ['domain1']
algos = ['algo1', 'algo2', 'algo3', 'algo4']
paths = ["./task"]
task = 'compare'
domain_algo_plot, algo_domain_path, retrived_algos = get_parsed_dict(paths)

for id, algo in enumerate(retrived_algos):
    if algo in algos:  # remove null results algo
        retrived_algos.pop(id)
algos.extend(retrived_algos)  # include more algo and keep the algo index not change

# COLORS = ['#77AC30', '#A56DB0', "#F0C04A", '#DE6C3A', '#2988C7', '#0000FF']
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']
MARKERS = ['o', '*', 's', '^']
LINES = ['-', '--', ':']
metrics = ["metric1"]

# Plot line chart
for domain in domains:
    plt.clf()
    plt.figure(figsize=(4.8, 4.2)) # width and height
    max_x_len = 0
    max_y_value = -1e6
    min_y_value = 1e6
    for algo in domain_algo_plot[domain]:
        data_path = algo_domain_path[algo + '-' + domain] # can  get multiple metrics result
        res_dict = get_results(data_path, metrics=metrics)
        single_metric_res = res_dict[metrics[0]]
        results = smooth_results(single_metric_res)
        mean = np.mean(results, axis=1)
        std = np.std(results, axis=1)

        x_vals = np.arange(len(mean))  # x axis item interval

        color = COLORS[algos.index(algo)]
        marker = MARKERS[algos.index(algo) % 4 ]
        line = LINES[algos.index(algo)//3]
        marker_num = 8
        makerevery = x_vals[-1] // marker_num
        plt.plot(x_vals, mean, label=algo, marker=marker, markevery=makerevery, markerfacecolor='none', markersize=5.5, markeredgewidth=1.5, color=color, linestyle=line)
        plt.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.3)

        max_x_len = max(max_x_len, x_vals[-1])
        max_y_value = max(max_y_value, np.max(mean+std))
        min_y_value = max(min_y_value, np.min(mean + std))

    # Plot misc
    plt.ylabel('reward')
    plt.xlabel('million steps')

    x_tick_interval = max_x_len//3  # just want five ticks, let it be max_len//5
    x_ticks = np.arange(0, max_x_len+1, x_tick_interval)
    plt.xticks(x_ticks, np.around(x_ticks/x_tick_interval, 1))
    plt.xlim(0, x_ticks[-1]+1)
    plt.ylim(np.ceil(min_y_value), np.ceil(max_y_value))
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    plt.grid(True, linestyle='-', alpha=0.5)
    lgd = plt.legend(loc='lower right', bbox_to_anchor=(1, 0), ncol=2, fancybox=False, framealpha=1, edgecolor='black', prop={'size': 8})
    lgd.get_frame().set_alpha(None)
    lgd.get_frame().set_facecolor((0, 0, 0, 0))
    plt.tight_layout()

    plt.savefig('./png/'+task+'_'+domain+'.png', bbox_inches='tight',  dpi=300)
