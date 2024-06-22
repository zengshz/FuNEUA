import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionModel import UserAttentionModel


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
        cpu_weight = self.get_fuzzy_weight(torch.mean(fuzzy_outputs[:, 0]), torch.std(fuzzy_outputs[:, 0]) if fuzzy_outputs[:, 0].numel() > 1 else 0)
        mem_weight = self.get_fuzzy_weight(torch.mean(fuzzy_outputs[:, 1]), torch.std(fuzzy_outputs[:, 1]) if fuzzy_outputs[:, 1].numel() > 1 else 0)
        bw_weight = self.get_fuzzy_weight(torch.mean(fuzzy_outputs[:, 2]), torch.std(fuzzy_outputs[:, 2]) if fuzzy_outputs[:, 2].numel() > 1 else 0)
        storage_weight = self.get_fuzzy_weight(torch.mean(fuzzy_outputs[:, 3]), torch.std(fuzzy_outputs[:, 3]) if fuzzy_outputs[:, 3].numel() > 1 else 0)
        return (cpu_weight + mem_weight + bw_weight + storage_weight) / 4.0



# 输出层
class DefuzzyLayer(nn.Module):
    def __init__(self):
        super(DefuzzyLayer, self).__init__()

    def forward(self, B, C, atte, total_weight):
        scores = []
        max_c, min_c = max(C), min(C)
        max_b, min_b = max(B), min(B)
        for i in range(len(C)):
            ci = (C[i] - min_c) / (max_c - min_c) if max_c - min_c != 0 else 0
            bi = (B[i] - min_b) / (max_b - min_b) if max_b - min_b != 0 else 0
            scores.append((total_weight + 0.3) * bi + (1 - total_weight + 0.55) * ci + atte[i])
        return scores


class FuzzyUserAllocator(nn.Module):
    def __init__(self, user_size, n_heads, embed_dim, device):
        super(FuzzyUserAllocator, self).__init__()
        self.device = device
        self.user_attention_net = UserAttentionModel(n_heads, embed_dim, user_size).to(device)
        self.fuzzy_layer = FuzzyLayer()
        self.inference_layer = InferenceLayer()
        self.defuzzy_layer = DefuzzyLayer()

    def choose_server(self, users_workload, masks, server_usage, tmp_server_capacity, servers, server_scores):
        final_server_ids = []
        servers_initial_capacities = []
        server_remain_capacity = []
        B = []
        C = []
        atte = []
        for server_id in range(masks.size(0)):
            if (masks[server_id]):
                if torch.all(tmp_server_capacity[server_id] >= users_workload):
                    final_server_ids.append(server_id)
                    servers_initial_capacities.append(servers[server_id][3:])
                    server_remain_capacity.append(servers[server_id][3:] - tmp_server_capacity[server_id])
                    capacities_utilization = tmp_server_capacity[server_id] / servers[server_id][3:]
                    capacities_utilization = torch.mean(capacities_utilization)
                    B.append(capacities_utilization)
                    if server_usage[server_id] == 0:
                        c = 0
                    else:
                        c = 10
                    C.append(c)
                    atte.append(server_scores[server_id])
        if final_server_ids:
            servers_initial_capacities = torch.stack(servers_initial_capacities)
            server_remain_capacity = torch.stack(server_remain_capacity)
            total_utilization_fuzzy = self.fuzzy_layer.forward(servers_initial_capacities, server_remain_capacity)
            total_weight = self.inference_layer.forward(total_utilization_fuzzy)
            scores = self.defuzzy_layer.forward(B, C, atte, total_weight)
            return final_server_ids, torch.tensor(scores, device=self.device)
        else:
            return None, None

    @staticmethod
    def compute_loss(user_allocated_prop_tensor, server_used_prop_tensor, context_vector, server_context_vector):
        # Create tensors once
        user_allocated_tensor = torch.tensor([user_allocated_prop_tensor], device=context_vector.device)
        server_used_tensor = torch.tensor([server_used_prop_tensor], device=context_vector.device)

        # Calculate user and server loss on context_vector.device
        user_loss = F.binary_cross_entropy_with_logits(user_allocated_tensor,
                                                       torch.tensor([1.0], device=context_vector.device))
        server_loss = F.binary_cross_entropy_with_logits(server_used_tensor,
                                                         torch.tensor([0.0], device=context_vector.device))

        # Calculate server user and server loss on server_context_vector.device
        user_loss_server = F.binary_cross_entropy_with_logits(user_allocated_tensor.to(server_context_vector.device),
                                                              torch.tensor([1.0], device=server_context_vector.device))
        server_loss_server = F.binary_cross_entropy_with_logits(server_used_tensor.to(server_context_vector.device),
                                                                torch.tensor([0.0],
                                                                             device=server_context_vector.device))

        total_loss = user_loss + server_loss + user_loss_server + server_loss_server

        # Calculate softmax once
        user_context_vector_normalized = F.softmax(context_vector, dim=0)
        server_context_vector_normalized = F.softmax(server_context_vector, dim=0)

        # Calculate weighted total loss
        weighted_total_loss = torch.sum(user_context_vector_normalized * total_loss) + torch.sum(
            server_context_vector_normalized * total_loss)
        return weighted_total_loss


    def forward(self, servers, users, masks):
        users_need = users[:, 2:]
        context_vector = self.user_attention_net(users_need)
        user_scores = [torch.mean(features) for features in context_vector]
        user_scores = torch.stack(user_scores)
        sorted_users = torch.argsort(user_scores, descending=True)  # 降序排序

        server_context_vector = self.user_attention_net(servers[:, 3:])
        server_scores = [torch.mean(features) for features in server_context_vector]

        tmp_server_capacity = servers[:, 3:].clone()
        server_usage = torch.zeros(servers.size(0), dtype=torch.float32, device=self.device)
        allocations = torch.full((users.size(0),), -1, dtype=torch.float, device=self.device)

        for user_id in range(users.size(0)):
            workload = users[sorted_users[user_id]][2:6]
            final_server_ids, scores = self.choose_server(workload, masks[sorted_users[user_id]], server_usage,
                                                          tmp_server_capacity, servers, server_scores)
            if final_server_ids:
                chosen_server_idx = torch.argmax(scores).item()
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

        loss = self.compute_loss(user_allocated_prop, server_used_prop, context_vector, server_context_vector)

        return loss, allocations, server_usage, user_allocated_prop, server_used_prop
