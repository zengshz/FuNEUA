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

def greedy_allocate(servers, users, user_masks):
    user_num = len(users)
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)
    user_allocate_list = [-1] * user_num
    server_allocate_user_num = [0] * server_num

    for user_id in range(user_num):
        workload = users[user_id][2:]

        max_remain_capacity = -1
        final_server_id = -1
        for server_id in user_within_servers[user_id]:
            capacity = servers[:, 3:][server_id]
            if np.all(capacity >= workload):
                remain_capacity = sum(capacity)
                if remain_capacity > max_remain_capacity:
                    max_remain_capacity = remain_capacity
                    final_server_id = server_id
        if final_server_id >= 0:
            servers[:, 3:][final_server_id] -= workload
            user_allocate_list[user_id] = final_server_id
            server_allocate_user_num[final_server_id] += 1


    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    used_server_num = server_num - server_allocate_user_num.count(0)
    server_used_prop = used_server_num / server_num

    return user_allocate_list, server_allocate_user_num, user_allocated_prop, server_used_prop
