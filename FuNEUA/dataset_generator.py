import math
import random
import numpy as np
import os

from torch.utils.data import Dataset
from tqdm import tqdm
import torch

workload_list = [
    np.array([1, 2, 1, 2]),
    np.array([2, 3, 3, 4]),
    np.array([5, 7, 6, 6])
]

def get_within_servers(user_list, server_list, x_start, x_end, y_start, y_end):
    users_masks = np.zeros((len(user_list), len(server_list)), dtype=bool)

    def calc_user_within(calc_user, index):
        flag = False
        for j in range(len(server_list)):
            if (np.linalg.norm(calc_user[:2] - server_list[j][:2]) <= server_list[j][2]):
                users_masks[index, j] = 1
                flag = True
        return flag

    for i in range(len(user_list)):
        user = user_list[i]
        user_within = calc_user_within(user, i)
        while not user_within:
            user[0] = random.random() * (x_end - x_start) + x_start
            user[1] = random.random() * (y_end - y_start) + y_start
            user_within = calc_user_within(user, i)
    return user_list, users_masks


def miller_to_xy(lon, lat):
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x = lon * math.pi / 180
    y = lat * math.pi / 180
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return x, y


def get_all_server_xy():
    server_list = []
    file = open("site-optus-melbCBD.csv", 'r')
    file.readline().strip()
    lines = file.readlines()
    for i in range(len(lines)):
        result = lines[i].split(',')
        server_mes = (float(result[2]), float(result[1]))
        x, y = miller_to_xy(*server_mes)
        server_list.append([x, y])
    file.close()

    server_list = np.array(server_list)
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy

    angle = 13
    for xy in server_list:
        x = xy[0] * math.cos(math.pi / 180 * angle) - xy[1] * math.sin(math.pi / 180 * angle)
        y = xy[0] * math.sin(math.pi / 180 * angle) + xy[1] * math.cos(math.pi / 180 * angle)
        xy[0] = x
        xy[1] = y

    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy

    for xy in server_list:
        xy[0] = xy[0] - xy[1] * math.tan(math.pi / 180 * 15)

    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    server_list /= 100
    return server_list


def init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop, min_cov=1, max_cov=1.5, miu=35, sigma=10):
    server_xy_list = get_all_server_xy()
    max_x_y = np.max(server_xy_list, axis=0)
    max_x = max_x_y[0]
    max_y = max_x_y[1]
    x_start = max_x * x_start_prop
    x_end = max_x * x_end_prop
    y_start = max_y * y_start_prop
    y_end = max_y * y_end_prop

    filter_server = [x_start <= server[0] <= x_end and y_start <= server[1] <= y_end for server in server_xy_list]
    server_xy_list = server_xy_list[filter_server]
    min_xy = np.min(server_xy_list, axis=0)
    server_xy_list = server_xy_list - min_xy + max_cov
    server_cov_list = np.random.uniform(min_cov, max_cov, (len(server_xy_list), 1))
    server_capacity_list = np.random.normal(miu, sigma, size=(len(server_xy_list), 4))
    server_list = np.concatenate((server_xy_list, server_cov_list, server_capacity_list), axis=1)
    return server_list


def init_users_list_by_server(server_list, data_num, user_num, max_cov=1.5):
    max_server = np.max(server_list, axis=0)
    max_x = max_server[0] + max_cov
    max_y = max_server[1] + max_cov
    min_server = np.min(server_list, axis=0)
    min_x = min_server[0] - max_cov
    min_y = min_server[1] - max_cov

    users_list = []
    users_masks_list = []
    for _ in tqdm(range(data_num)):
        user_x_list = np.random.uniform(min_x, max_x, (user_num, 1))
        user_y_list = np.random.uniform(min_y, max_y, (user_num, 1))
        user_load_list = np.array([random.choice(workload_list) for _ in range(user_num)])
        user_list = np.concatenate((user_x_list, user_y_list, user_load_list), axis=1)
        user_list, users_masks = get_within_servers(user_list, server_list, min_x, max_x, min_y, max_y)
        users_list.append(user_list)
        users_masks_list.append(users_masks)

    return {"users_list": users_list, "users_masks_list": users_masks_list}


class EuaDataset(Dataset):
    def __init__(self, servers, users_list, users_masks_list, device):
        self.servers, self.users_list, self.users_masks_list = servers, users_list, users_masks_list
        self.servers_tensor = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, index):
        user_seq = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        mask_seq = torch.tensor(self.users_masks_list[index], dtype=torch.bool, device=self.device)
        return self.servers_tensor, user_seq, mask_seq

def get_dataset(x_end, y_end, miu, sigma, user_num, data_size: {}, min_cov, max_cov, device, dir_name):
    dataset_dir_name = os.path.join(dir_name, "dataset/server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma))
    server_file_name = "server_" + str(x_end) + "_" + str(y_end) + "_miu_" + str(miu) + "_sigma_" + str(sigma)
    server_path = os.path.join(dataset_dir_name, server_file_name) + '.npy'

    if os.path.exists(server_path):
        servers = np.load(server_path)
    else:
        os.makedirs(dataset_dir_name, exist_ok=True)
        servers = init_server(0, x_end, 0, y_end, min_cov, max_cov, miu, sigma)
        np.save(server_path, servers)

    set_types = data_size.keys()
    datasets = {}
    for set_type in set_types:
        if set_type not in ('train', 'valid', 'test'):
            raise NotImplementedError
        filename = set_type + "_user_" + str(user_num) + "_size_" + str(data_size[set_type])
        path = os.path.join(dataset_dir_name, filename) + '.npz'
        if os.path.exists(path):
            print("loading", set_type)
            data = np.load(path)
        else:
            print(set_type, "Generating", path)
            data = init_users_list_by_server(servers, data_size[set_type], user_num, max_cov)
            np.savez_compressed(path,**data)
        datasets[set_type] = EuaDataset(servers, **data, device=device)
    return datasets