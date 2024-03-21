# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConstruction(nn.Module):
    def __init__(self, patch_size, hop_distance):
        super().__init__()
        self.patch_size = patch_size
        self.hop_distance = hop_distance

    def forward(self, x):
        patches = self.patchify(x)
        A = self.create_adjacency_matrix(patches)
        return A

    def patchify(self, x):
        x = x.unsqueeze(0)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.view(-1, self.patch_size, self.patch_size)
        return patches

    def create_adjacency_matrix(self, patches):
        num_patches = patches.shape[0]
        A = torch.zeros(num_patches, num_patches)
        for i in range(num_patches):
            for j in range(num_patches):
                if self.distance(patches[i], patches[j]) <= self.hop_distance:
                    A[i, j] = 1
        return A

    def distance(self, patch1, patch2):
        return torch.norm(patch1 - patch2, dim=1)
