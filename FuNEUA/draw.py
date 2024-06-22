import sys
import yaml
import os
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from dataset_generator import get_dataset
from torch.utils.data import DataLoader
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def draw():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)[
            'data']
    script_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(script_path)
    user_num = config['user_num']
    x_end = config['x_end']
    y_end = config['y_end']
    min_cov = config['min_cov']
    max_cov = config['max_cov']
    miu = config['miu']
    sigma = config['sigma']
    data_size = config['data_size']
    batch_size = config['batch_size']
    device = torch.cuda.current_device()

    dataset = get_dataset(x_end, y_end, miu, sigma, user_num, data_size, min_cov, max_cov, device, dir_name)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=batch_size, shuffle=False)
    for _, (test_server_seq, test_user_seq, test_masks) in enumerate(test_loader):
        for batch_idx in range(len(test_server_seq)):
            print('Number of users：' + str(len(test_user_seq[batch_idx])) + '; Number of servers：' + str(
                len(test_server_seq[batch_idx])))
            sns.set_theme(style='whitegrid')
            fig, ax = plt.subplots(figsize=(10, 8))
            for server in test_server_seq[batch_idx].cpu().numpy():
                circle = Circle((server[0], server[1]), server[2], alpha=0.3, edgecolor='blue', linestyle='--',
                                linewidth=2, facecolor='none')
                ax.add_patch(circle)
                sns.scatterplot(x=[server[0]], y=[server[1]], marker='o', color='blue', s=60, ax=ax, label=None)

            for user in test_user_seq[batch_idx].cpu().numpy():
                sns.scatterplot(x=[user[0]], y=[user[1]], color='red', s=20, ax=ax, label=None)

            ax.set_aspect('equal')

            ax.set_xlabel('x(m)')
            ax.set_ylabel('y(m)')

            def times_hundred(x):
                return int(x * 100)

            formatter = FuncFormatter(times_hundred)

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            sns.despine(ax=ax, offset=10, trim=True)
            plt.tight_layout()

            ax.legend(
                handles=[plt.Line2D([], [], color='blue', marker='o', markersize=10, linestyle='None', label='server'),
                         plt.Line2D([], [], color='red', marker='o', markersize=10, linestyle='None', label='user')],
                loc='upper right')
            plt.savefig('Example.svg')
            plt.show()


            sys.exit()

if __name__ == '__main__':
    draw()