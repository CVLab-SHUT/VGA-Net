# -*- coding: utf-8 -*-

import torch.nn as nn
import GCN

class GraphFeatureExtraction(nn.Module):
    def __init__(self, dropout_rate, feature_dim, num_heads, num_layers):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Module GAT for graph-level feature extraction
        self.graph_conv = GCN(feature_dim, num_heads, num_layers)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, A, node_features):
        # Extract graph-level features
        graph_features = self.graph_conv(node_features, A)

        # Apply Dropout
        graph_features = self.dropout(graph_features)

        return graph_features
