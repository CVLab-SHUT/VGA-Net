# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers):
        super(GCN, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # GAT layers
        self.gat_layers = torch.nn.ModuleList([GATConv(feature_dim, num_heads) for _ in range(num_layers)])

    def forward(self, inputs, adj):
        # Applying GAT layers sequentially
        for gat_layer in self.gat_layers:
            inputs = gat_layer(inputs, adj)

        # GCN output
        return inputs

class GATConv(torch.nn.Module):
    def __init__(self, feature_dim, num_heads, **kwargs):
        super(GATConv, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Weights and biases for attention function
        self.W = torch.nn.Linear(feature_dim, feature_dim * num_heads, bias=False)
        self.a = torch.nn.Linear(1, 1, bias=True)

        # Activation for attention and final output
        self.activation = F.leaky_relu

    def forward(self, inputs, adj):
        # Wh and Ws: node features (shape: [num_heads, feature_dim])
        Wh, Ws = self.W(inputs).chunk(self.num_heads, dim=-1)
        bias = self.a(torch.ones_like(Wh[:, :1])).view(-1)

        # Calculating attention score (a)
        a = torch.matmul(torch.tanh(Wh + Ws + bias), Wh.transpose(-1, -2))
        a = F.softmax(a, dim=-1)

        # Weighted sum of features
        out = torch.matmul(a, Wh)

        # Residual connection and layer normalization
        out = out + inputs
        out = F.batch_norm(out, training=self.training)
        out = self.activation(out)

        return out
