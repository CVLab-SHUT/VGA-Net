# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AB_FFMModule(nn.Module):
    def __init__(self):
        super().__init__()

        # BConvLSTM

        self.bconvlstm = BConvLSTM(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        #  Dual attention network

        self.attention = DualAttentionModule()

    def forward(self, x):
        # BConvLSTM

        features = self.bconvlstm(x)

        # Dual attention network

        features = self.attention(features)

        return features
class BConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        # دو ConvLSTM در جهت های مخالف

        self.forward_lstm = nn.ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.backward_lstm = nn.ConvLSTM(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # عبور از دو ConvLSTM

        forward_out, _ = self.forward_lstm(x)
        backward_out, _ = self.backward_lstm(x.flip(0))

        # ترکیب خروجی ها

        features = torch.cat((forward_out, backward_out), dim=1)

        return features



class DualAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Pixel-wise attention module

        self.pam = PixelAttentionModule()

        # Channel-wise attention module

        self.cam = ChannelAttentionModule()

    def forward(self, x):
        # Pixel-wise attention

        pixel_weights = self.pam(x)

        # Channel-wise attention

        channel_weights = self.cam(x)

        # ترکیب دو نوع توجه

        features = x * pixel_weights * channel_weights

        return features


class PixelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layer and max pooling

        self.conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Attention map

        attention_map = self.conv(x)
        attention_map = self.pool(attention_map)

        # Pixel-wise weights

        pixel_weights = torch.sigmoid(attention_map)

        return pixel_weights


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layer and global average pooling

        self.conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # Attention map

        attention_map = self.conv(x)
        attention_map = self.pool(attention_map)

        # Channel-wise weights

        channel_weights = torch.sigmoid(attention_map)

        return channel_weights