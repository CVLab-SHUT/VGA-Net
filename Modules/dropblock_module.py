# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Dropblock(nn.Module):
    def __init__(self, block_size, drop_prob):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            # Generate mask
            mask = self._generate_block_mask(x)

            # Apply mask
            x = x * mask

            # Normalize to compensate for dropped elements
            x = x * (mask.numel() / mask.sum())
        return x

    def _generate_block_mask(self, x):
        mask = torch.zeros_like(x)
        _, _, h, w = x.size()
        for i in range(0, h - self.block_size + 1):
            for j in range(0, w - self.block_size + 1):
                # Drop block with probability drop_prob
                if torch.rand(1).item() < self.drop_prob:
                    mask[:, :, i:i+self.block_size, j:j+self.block_size] = 0
        return mask

# Example usage
dropblock = Dropblock(block_size=7, drop_prob=0.15)
