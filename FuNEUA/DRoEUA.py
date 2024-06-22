import copy
import numpy as np

def mask_trans_to_list(user_masks, server_num):
    x = []
    user_masks = user_masks.astype(bool)
    server_arrange = np.arange(server_num)
    for i in range(len(user_masks)):
        mask = user_masks[i]
        y = server_arrange[mask]
        x.append(y.tolist())
    return x

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1
omega_dic = {'ML': {"SL": EL, "SM": VL, "SH": VL},
             'MM': {"SL": M, "SM": L, "SH": VL},
             'MH': {"SL": EH, "SM": VH, "SH": H}}
gamma = 1.5

def get_fuzzy_weight(mu, std):
    if mu <= 0.09:
        a = 'ML'
    elif 0.09 < mu <= 0.22:
        a = 'MM'
    else:
        a = 'MH'
    if std <= 0.03:
        b = 'SL'
    elif 0.03 < std <= 0.12:
        b = 'SM'
    else:
        b = 'SH'
    return omega_dic[a][b]


def fuzzy_allocate(servers, users, user_masks):
    user_num = len(users)
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)
    user_allocate_list = [-1] * user_num
    server_allocate_user_num = [0] * server_num
    tmp_server_capacity = np.array(copy.deepcopy(servers[:, 3:]))

    for user_id in range(user_num):
        workload = users[user_id][2:]
        final_server_ids = []
        C = []
        B = []
        for server_id in user_within_servers[user_id]:
            capacity = tmp_server_capacity[server_id]
            if np.all(capacity >= workload):
                final_server_ids.append(server_id)
                zi = 0 if server_allocate_user_num[server_id] == 0 else 10
                t = 0
                vj = 10
                c = abs(zi - (t + vj))
                if zi < t + vj:
                    c = c * gamma
                C.append(c)
                b = np.mean(1 - tmp_server_capacity[server_id, :4] / servers[server_id, 3:7])
                B.append(b)
        if final_server_ids:
            capacity_used_props = np.zeros(server_num)
            for server_id in range(server_num):
                capacity_used_props[server_id] = np.mean(1- tmp_server_capacity[server_id, :4] / servers[server_id, 3:7])
            mu = np.mean(capacity_used_props)
            std = np.std(capacity_used_props)
            omega_j = get_fuzzy_weight(mu, std)
            max_c, min_c = max(C), min(C)
            max_b, min_b = max(B), min(B)
            S = []
            for i in range(len(C)):
                ci = (C[i] - min_c) / (max_c - min_c) if max_c - min_c != 0 else 0
                bi = (B[i] - min_b) / (max_b - min_b) if max_b - min_b != 0 else 0
                S.append((omega_j) * ci + (1 - omega_j) * bi)
            final_server_id = final_server_ids[np.argmin(np.array(S))]
            tmp_server_capacity[final_server_id] -= workload  #
            user_allocate_list[user_id] = final_server_id
            server_allocate_user_num[final_server_id] += 1


    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    used_server_num = server_num - server_allocate_user_num.count(0)
    server_used_prop = used_server_num / server_num

    return user_allocate_list, server_allocate_user_num, user_allocated_prop, server_used_prop
