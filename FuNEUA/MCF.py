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

def mcf_allocate(servers, users_original, masks_original):
    x = zip(users_original,masks_original)
    x = list(x)
    x = sorted(x, key=lambda u: u[0][2])
    users, user_masks = zip(*x)
    users_sorted, user_masks_sorted = np.array(users, dtype=float), np.array(user_masks,dtype=bool)

    user_num = len(users_sorted)
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks_sorted, server_num)
    user_allocate_list = [-1] * user_num
    server_allocate_user_num = [0] * server_num

    for user_id in range(user_num):
        workload = users_sorted[user_id][2:]
        this_user_s_active_server = []
        other_servers = []

        for server_id in user_within_servers[user_id]:
            if server_allocate_user_num[server_id] > 0:
                this_user_s_active_server.append(server_id)
            else:
                other_servers.append(server_id)

        max_remain_capacity = -1
        final_server_id = -1
        for server_id in this_user_s_active_server:
            capacity = servers[:, 3:][server_id]
            if np.all(capacity >= workload):
                remain_capacity = sum(capacity) - sum(workload)
                if remain_capacity > max_remain_capacity:
                    max_remain_capacity = remain_capacity
                    final_server_id = server_id
        if final_server_id != -1:
            servers[:, 3:][final_server_id] -= workload
            user_allocate_list[user_id] = final_server_id
            server_allocate_user_num[final_server_id] += 1

        else:
            for server_id in other_servers:
                capacity = servers[:, 3:][server_id]
                if np.all(capacity >= workload):
                    remain_capacity = sum(capacity) - sum(workload)
                    if remain_capacity > max_remain_capacity:
                        max_remain_capacity = remain_capacity
                        final_server_id = server_id
            if final_server_id != -1:
                servers[:, 3:][final_server_id] -= workload
                user_allocate_list[user_id] = final_server_id
                server_allocate_user_num[final_server_id] += 1


    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    used_server_num = server_num - server_allocate_user_num.count(0)
    server_used_prop = used_server_num / server_num

    return users_sorted, user_allocate_list, user_allocated_prop, server_used_prop
