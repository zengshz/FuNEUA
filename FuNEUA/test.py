import time
import yaml
import os
import torch
import pandas as pd
from dataset_generator import get_dataset
from torch.utils.data import DataLoader
from FuNEUA import FuzzyUserAllocator
from DRoEUA import fuzzy_allocate
from Greedy import greedy_allocate
from MCF import mcf_allocate
from Random import random_allocate


def test():
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
    user_size = config['user_size']
    n_heads = config['n_heads']
    embed_dim = config['embed_dim']
    device = torch.cuda.current_device()

    # user_num = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # x_end = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    miu = [20, 30, 40, 50, 60, 70, 80]

    for change in miu:
        dataset = get_dataset(x_end, y_end, change, sigma, user_num, data_size, min_cov, max_cov, device, dir_name)
        test_loader = DataLoader(dataset=dataset['test'], batch_size=batch_size, shuffle=False)

        # 模型
        model = FuzzyUserAllocator(user_size, n_heads, embed_dim, device)

        model.load_state_dict(torch.load(dir_name + "/model/06211613best_model.pth"))
        model.eval()
        experimental_results = pd.DataFrame()
        with torch.no_grad():
            for _, (test_server_seq, test_user_seq, test_masks) in enumerate(test_loader):
                for batch_idx in range(len(test_server_seq)):
                    loss, allocations, server_usage, fnn_user_allocated_prop, fnn_server_used_prop = model(
                        test_server_seq[batch_idx], test_user_seq[batch_idx], test_masks[batch_idx])


                    random_user_allocate_list, random_server_allocate_user_num, random_user_allocated_prop, random_server_used_prop = \
                        random_allocate(test_server_seq[batch_idx].cpu().numpy(),
                                        test_user_seq[batch_idx].cpu().numpy(),
                                        test_masks[batch_idx].cpu().numpy())


                    greedy_user_allocate_list, greedy_server_allocate_user_num, greedy_user_allocated_prop, greedy_server_used_prop = \
                        greedy_allocate(test_server_seq[batch_idx].cpu().numpy(),
                                        test_user_seq[batch_idx].cpu().numpy(),
                                        test_masks[batch_idx].cpu().numpy())


                    mcf_sorted_user_list, mcf_user_allocate_list, mcf_user_allocated_prop, mcf_server_used_prop = \
                        mcf_allocate(test_server_seq[batch_idx].cpu().numpy(), test_user_seq[batch_idx].cpu().numpy(),
                                     test_masks[batch_idx].cpu().numpy())


                    fuzzy_user_allocate_list, fuzzy_server_allocate_user_num, fuzzy_user_allocated_prop, fuzzy_server_used_prop = \
                        fuzzy_allocate(test_server_seq[batch_idx].cpu().numpy(),
                                       test_user_seq[batch_idx].cpu().numpy(),
                                       test_masks[batch_idx].cpu().numpy())


                    res = pd.DataFrame([[len(test_user_seq[batch_idx]), len(test_server_seq[batch_idx]),
                                         fnn_user_allocated_prop.item(), fnn_server_used_prop.item(),
                                         random_user_allocated_prop, random_server_used_prop, greedy_user_allocated_prop,
                                         greedy_server_used_prop, mcf_user_allocated_prop, mcf_server_used_prop,
                                         fuzzy_user_allocated_prop, fuzzy_server_used_prop]],
                                       columns=['num_users', 'num_servers', 'fnn_user',
                                                'fnn_server', 'random_user', 'random_server', 'greedy_user',
                                                'greedy_server',
                                                'mcf_user', 'mcf_server', 'fuzzy_user', 'fuzzy_server'])
                    experimental_results = pd.concat([experimental_results, res])
                fname = dir_name + "/result/"
                if not os.path.exists(fname):
                    os.makedirs(fname)
                fname = fname + time.strftime(
                    '%m%d%H%M', time.localtime(time.time())
                ) + '_' + str(user_num) + '_' + str(x_end) + '_' + str(y_end) + '_' + str(change) + 'result.csv'
                experimental_results.to_csv(fname)
                print("Write the result to :" + str(fname))

if __name__ == '__main__':
    test()
