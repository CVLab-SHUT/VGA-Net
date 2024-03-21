# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F
from hdc_module import HDCModule
from ab_ffm_module import AB_FFMModule
from GCN import GATConv 
from dropblock import Dropblock

class VGA_Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # HDC module
        self.hdc = HDCModule()

        # AB-FFM module
        self.ab_ffm = AB_FFMModule()

        # Dropblock method
        self.dropblock = Dropblock(block_size=7, drop_prob=0.15)

        # Encoder pathway
        self.encoder1 = self.hdc
        self.encoder2 = self.hdc
        self.encoder3 = self.hdc
        self.encoder4 = self.hdc

        # Decoder pathway
        self.decoder1 = self.hdc
        self.decoder2 = self.hdc
        self.decoder3 = self.hdc
        self.decoder4 = self.hdc

        # Output layer
        self.output = nn.Sigmoid()

        # Graph Attention Module
        self.gat = GATConv(feature_dim=512, num_heads=8)

    def forward(self, patches, A):
        # Encoder pathway
        x1 = self.encoder1(patches)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # HDC module
        x4 = self.hdc(x4)

        # Decoder pathway
        x3 = self.decoder1(x4)
        x2 = self.decoder2(x3)
        x1 = self.decoder3(x2)
        x = self.decoder4(x1)

        # Graph Attention Module
        A_out = self.gat(x, A)

        # Reshape A_out to match the feature map size
        A_out = F.interpolate(A_out, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Inject A_out into each skip connection
        x1 = x1 + A_out
        x2 = x2 + A_out
        x3 = x3 + A_out

        # AB-FFM module
        x = self.ab_ffm(x, x1, x2, x3)

        # Dropblock method
        x = self.dropblock(x)

        # Output layer
        x = self.output(x)

        return x
