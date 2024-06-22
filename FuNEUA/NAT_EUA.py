import torch
import torch.nn as nn
import torch.nn.functional as F

from AttentionModel import UserAttentionModel

class FuzzyUserAllocator_1(nn.Module):
    def __init__(self, user_size, n_heads, embed_dim, device):
        super(FuzzyUserAllocator_1, self).__init__()
        self.device = device
        self.user_attention_net = UserAttentionModel(n_heads, embed_dim, user_size).to(device)

    @staticmethod
    def compute_loss(user_allocated_prop_tensor, server_used_prop_tensor, context_vector, server_context_vector):
        user_loss = F.binary_cross_entropy_with_logits(
            torch.tensor([user_allocated_prop_tensor], device=context_vector.device),
            torch.tensor([1.0], device=context_vector.device)
        )
        server_loss = F.binary_cross_entropy_with_logits(
            torch.tensor([server_used_prop_tensor], device=context_vector.device),
            torch.tensor([0.0], device=context_vector.device)
        )

        server_user_loss = F.binary_cross_entropy_with_logits(
            torch.tensor([user_allocated_prop_tensor], device=server_context_vector.device),
            torch.tensor([1.0], device=server_context_vector.device)
        )
        server_server_loss = F.binary_cross_entropy_with_logits(
            torch.tensor([server_used_prop_tensor], device=server_context_vector.device),
            torch.tensor([0.0], device=server_context_vector.device)
        )

        # Combine the losses into a total_loss tensor
        total_loss = user_loss + server_loss + server_user_loss + server_server_loss

        # Normalize the combined context vector
        user_context_vector_normalized = F.softmax(context_vector, dim=0)
        server_context_vector_normalized = F.softmax(server_context_vector, dim=0)

        # Compute the weighted total loss
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

            max_scores = -1
            final_server_id = -1
            for server_id in range(masks[sorted_users[user_id]].size(0)):
                if (masks[sorted_users[user_id]][server_id]):
                    capacity = tmp_server_capacity[server_id]
                    if torch.all(capacity >= workload):
                        if server_scores[server_id] > max_scores:
                            max_scores = server_scores[server_id]
                            final_server_id = server_id

            if final_server_id != -1:
                tmp_server_capacity[final_server_id] -= workload
                allocations[sorted_users[user_id]] = final_server_id
                server_usage[final_server_id] += 1.0

        allocated_user_num = torch.sum(allocations != -1)
        user_allocated_prop = allocated_user_num / users.size(0) if users.size(0) > 0 else torch.tensor(0.0,
                                                                                                        device=self.device)
        used_server_num = servers.size(0) - (server_usage.numel() - torch.count_nonzero(server_usage))
        server_used_prop = used_server_num / servers.size(0) if servers.size(0) > 0 else torch.tensor(0.0,
                                                                                                      device=self.device)

        loss = self.compute_loss(user_allocated_prop, server_used_prop, context_vector, server_context_vector)

        return loss, allocations, server_usage, user_allocated_prop, server_used_prop
