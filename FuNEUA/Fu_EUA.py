import torch
import torch.nn as nn

# 模糊层
class FuzzyLayer(nn.Module):
    def __init__(self):
        super(FuzzyLayer, self).__init__()

    @staticmethod
    def triangular_fuzzification(x, low, mid, high):
        fuzzified_x = torch.where(x <= low, torch.tensor(0.0, device=x.device),
                                  torch.where((low < x) & (x <= mid),
                                              (x - low) / (mid - low),
                                              torch.where((mid < x) & (x <= high),
                                                          (high - x) / (high - mid),
                                                          torch.tensor(0.0, device=x.device))))
        return fuzzified_x

    def forward(self, servers, rem_server):
        if rem_server.size(0) == 0:
            return torch.zeros(1, 4, device=servers.device)

        # Calculate the remaining resource ratios
        cpu_ratio = rem_server[:, 0] / servers[:, 0]
        mem_ratio = rem_server[:, 1] / servers[:, 1]
        bw_ratio = rem_server[:, 2] / servers[:, 2]
        storage_ratio = rem_server[:, 3] / servers[:, 3]

        # Fuzzification based on remaining resource ratios
        cpu_fuzzy = self.triangular_fuzzification(cpu_ratio, 0.0, 0.4, 0.8)
        mem_fuzzy = self.triangular_fuzzification(mem_ratio, 0.0, 0.5, 0.8)
        bw_fuzzy = self.triangular_fuzzification(bw_ratio, 0.0, 0.3, 0.7)
        storage_fuzzy = self.triangular_fuzzification(storage_ratio, 0.0, 0.5, 0.8)

        return torch.stack([cpu_fuzzy, mem_fuzzy, bw_fuzzy, storage_fuzzy], dim=1)


# 推理层
class InferenceLayer(nn.Module):
    def __init__(self):
        super(InferenceLayer, self).__init__()

    @staticmethod
    def get_fuzzy_weight(mu, std):
        if mu <= 0.2:
            return 0.9 if std <= 0.1 else (0.8 if std <= 0.3 else 0.6)
        elif mu <= 0.5:
            return 0.6 if std <= 0.1 else (0.5 if std <= 0.3 else 0.4)
        else:
            return 0.4 if std <= 0.1 else (0.2 if std <= 0.3 else 0.1)

    def forward(self, fuzzy_outputs):
        mean = torch.mean(fuzzy_outputs)
        std = torch.std(fuzzy_outputs)
        return self.get_fuzzy_weight(mean, std)



# 输出层
class DefuzzyLayer(nn.Module):
    def __init__(self):
        super(DefuzzyLayer, self).__init__()

    def forward(self, B, C, total_weight):
        scores = []
        max_c, min_c = max(C), min(C)
        max_b, min_b = max(B), min(B)
        for i in range(len(C)):
            ci = (C[i] - min_c) / (max_c - min_c) if max_c - min_c != 0 else 0
            bi = (B[i] - min_b) / (max_b - min_b) if max_b - min_b != 0 else 0
            scores.append((total_weight + 0.3) * bi + (1 - total_weight + 0.55) * ci)
        return scores


class FuzzyUserAllocator_2(nn.Module):
    def __init__(self, device):
        super(FuzzyUserAllocator_2, self).__init__()
        self.device = device

        self.fuzzy_layer = FuzzyLayer()
        self.inference_layer = InferenceLayer()
        self.defuzzy_layer = DefuzzyLayer()

    def choose_server(self, users_workload, masks, server_usage, tmp_server_capacity, servers):
        final_server_ids = []
        servers_initial_capacities = []
        server_remain_capacity = []
        B = []
        C = []
        for server_id in range(masks.size(0)):
            if (masks[server_id]):
                if torch.all(tmp_server_capacity[server_id] >= users_workload):
                    final_server_ids.append(server_id)
                    servers_initial_capacities.append(servers[server_id][3:])
                    server_remain_capacity.append(servers[server_id][3:] - tmp_server_capacity[server_id])
                    capacities_utilization = tmp_server_capacity[server_id] / servers[server_id][3:]
                    capacities_utilization = torch.mean(capacities_utilization)
                    B.append(1 - capacities_utilization)
                    if server_usage[server_id] == 0:
                        c = 10
                    else:
                        c = 0
                    C.append(c)
        if final_server_ids:
            servers_initial_capacities = torch.stack(servers_initial_capacities)
            server_remain_capacity = torch.stack(server_remain_capacity)
            total_utilization_fuzzy = self.fuzzy_layer.forward(servers_initial_capacities, server_remain_capacity)
            total_weight = self.inference_layer.forward(total_utilization_fuzzy)
            scores = self.defuzzy_layer.forward(B, C, total_weight)
            return final_server_ids, torch.tensor(scores, device=self.device)
        else:
            return None, None

    def forward(self, servers, users, masks):
        _, sorted_users = torch.sort(users[:, 2])

        tmp_server_capacity = servers[:, 3:].clone()
        server_usage = torch.zeros(servers.size(0), dtype=torch.float32, device=self.device)
        allocations = torch.full((users.size(0),), -1, dtype=torch.float, device=self.device)

        for user_id in range(users.size(0)):
            workload = users[sorted_users[user_id]][2:6]
            final_server_ids, scores = self.choose_server(workload, masks[sorted_users[user_id]], server_usage,
                                                          tmp_server_capacity, servers)
            if final_server_ids:
                chosen_server_idx = torch.argmin(scores).item()
                chosen_server = final_server_ids[chosen_server_idx]
                allocations[sorted_users[user_id]] = chosen_server
                tmp_server_capacity[chosen_server] -= workload
                server_usage[chosen_server] += 1.0

        allocated_user_num = torch.sum(allocations != -1)
        user_allocated_prop = allocated_user_num / users.size(0) if users.size(0) > 0 else torch.tensor(0.0,
                                                                                                        device=self.device)
        used_server_num = servers.size(0) - (server_usage.numel() - torch.count_nonzero(server_usage))
        server_used_prop = used_server_num / servers.size(0) if servers.size(0) > 0 else torch.tensor(0.0,
                                                                                                      device=self.device)

        return allocations, server_usage, user_allocated_prop, server_used_prop
