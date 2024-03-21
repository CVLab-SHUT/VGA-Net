# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DRIU(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Input image dimensions
        self.input_size = input_size

        # Number of output channels for specialized layers
        self.num_channels = 16

        # Base network architecture (VGG)
        self.base_network = VGG(input_size=input_size)

        # Specialized layers for blood vessels
        self.vessel_specialized_layers = nn.Sequential(
            nn.Conv2d(512, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Specialized layers for optic disc
        self.optic_disc_specialized_layers = nn.Sequential(
            nn.Conv2d(512, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Final layer for feature fusion
        self.final_layer = nn.Conv2d(self.num_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        # Pass through the base network
        feature_maps = self.base_network(x)

        # Extract pixel-level features
        vessel_features = self.vessel_specialized_layers(feature_maps[-4])
        optic_disc_features = self.optic_disc_specialized_layers(feature_maps[0])

        # Resize and combine features
        vessel_features = F.interpolate(vessel_features, size=self.input_size, mode="bilinear")
        optic_disc_features = F.interpolate(optic_disc_features, size=self.input_size, mode="bilinear")
        combined_features = torch.cat((vessel_features, optic_disc_features), dim=1)

        # Prediction
        segmentation_output = self.final_layer(combined_features)

        return segmentation_output

class VGG(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # VGG-16 architecture
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):
   
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        return [x]
