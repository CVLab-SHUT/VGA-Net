# -*- coding: utf-8 -*-

import torch.nn as nn
from graph_construction import GraphConstruction
from pixel_feature_extraction import DRIU
from graph_feature_extraction import GraphFeatureExtraction
from segmentation import VGA_Net
from patch_extraction import extract_random_patches

class FinalNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Input image dimensions
        self.input_size = (512, 512)

        # Graph parameters
        self.patch_size = 16
        self.hop_distance = 1

        # Dropout rate
        self.dropout_rate = 0.5

        # **Graph Construction Section**
        self.graph_construction = GraphConstruction(
            patch_size=self.patch_size, hop_distance=self.hop_distance
        )

        # **Pixel-level Feature Extraction Section**
        self.pixel_feature_extraction = DRIU(input_size=self.input_size)

        # **Graph-level Feature Extraction Section**
        self.graph_feature_extraction = GraphFeatureExtraction(
            dropout_rate=self.dropout_rate
        )

        # **Segmentation Section**
        self.segmentation = VGA_Net()

    def forward(self, x):
        
        j = self.extract_random_patches(x)
        # **Step 1: Graph Construction**
        A = self.graph_construction(x)

        # **Step 2: Pixel-level Feature Extraction**
        node_features = self.pixel_feature_extraction(x)

        # **Step 3: Graph-level Feature Extraction**
        graph_features = self.graph_feature_extraction(A, node_features)

        # **Step 4: Segmentation**
        segmentation_output = self.segmentation(j,graph_features)

        return segmentation_output
