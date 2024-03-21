# -*- coding: utf-8 -*-


import os
import cv2
import torch
from torch.utils.data import Dataset


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'masks')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.image_files[idx])
        mask_name = os.path.join(self.root_dir, 'masks', self.mask_files[idx])
        
        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return {'image': image, 'mask': mask}
