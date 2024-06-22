import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim
    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)
        return output, attn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        seq_len = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        attn_output, attn = self.attention(Q, K, V)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, self.embed_dim)
        output = self.out(attn_output)
        return output, attn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden=512):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, feed_forward_hidden)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feed_forward_hidden, embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class UserAttentionModel(nn.Module):
    def __init__(self, n_heads, embed_dim, feature_dim, feed_forward_hidden=512):
        super(UserAttentionModel, self).__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.attention_layer = MultiHeadAttentionLayer(n_heads, embed_dim)
        self.feed_forward = PositionwiseFeedforward(embed_dim, feed_forward_hidden)
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.input_proj(x)
        attn_output, attn = self.attention_layer(x)
        attn_output = x + attn_output
        attn_output = self.norm_layer(attn_output)
        ff_output = self.feed_forward(attn_output)
        ff_output = attn_output + ff_output
        output = self.norm_layer(ff_output)
        return output