import matplotlib.pyplot as plt
import numpy as np
import time
from utils import get_results, get_parsed_dict, smooth_results
import os

domains = ['domain1', 'domain2','domain3','domain4',]
algos = ['algo1', 'algo2', 'algo3', 'algo4']
paths = ["task"]
domain_algo_plot, algo_domain_path, retrived_algos = get_parsed_dict(paths)

for id, algo in enumerate(retrived_algos):
    if algo in algos:  # remove null results algo
        retrived_algos.pop(id)
algos.extend(retrived_algos)  # include more algo and keep the algo index not change
# COLORS = ['#77AC30', '#A56DB0', "#F0C04A", '#DE6C3A', '#2988C7', '#0000FF']
COLORS = ["#ccb974", '#8172b2', '#c44e52', '#55a868', '#4c72b0', '#0000FF']

metrics = ["metric2"]
figsize=()
ax = plt.figure().add_subplot(projection='3d')

max_len = 0
max_y_value = 0
for algo in algos:
    pre_domain = None
    color = COLORS[algos.index(algo)]
    for idx, domain in enumerate(domains):
        data_path = os.path.join(paths[0], algo, domain) # can  get multiple metrics result
        res_dict = get_results(data_path, metrics=metrics)
        single_metric_res = res_dict[metrics[0]]
        results = smooth_results(single_metric_res)
        mean = np.mean(results, axis=1)

        x_vals = np.arange(len(mean))  # x axis item interval

        if pre_domain:
            for yidx in range(0, len(mean), 5):
                shade = yidx//(len(mean)/6)
                ax.plot([pre_domain['x'], idx], [yidx,yidx], [pre_domain['z'][yidx],mean[yidx]], color=color, alpha=0.8-0.1*shade)

        pre_domain = {
            'x': idx,
            'z': mean
        }

        max_len = max(max_len, x_vals[-1])


ax.set_ylabel('million steps')
ax.set_xlabel('domain')
ax.set_zlabel('average return')
# ax.invert_yaxis()
# ax.legend()

# plt.tight_layout()
ax.view_init(elev=20, azim=45, roll=0)
plt.savefig('./png/3dlinechart.png',  dpi=300)
