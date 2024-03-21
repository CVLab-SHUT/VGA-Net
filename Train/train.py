# -*- coding: utf-8 -*-

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import transforms
from torchvision.transforms import ToTensor
from dataset import DRIVEDataset
from model import FinalNetwork
from utils import train_model

# Define the directory path for the DRIVE dataset
drive_dataset_dir = 'path/to/drive/dataset'

# Define transformations
transform = transforms.Compose([
    ToTensor(),
    # Add other transformations if needed
])

# Load the dataset
drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transform=None)

# Split the dataset into training and testing sets
train_size = int(0.7 * len(drive_dataset))
val_size = int(0.2 * len(drive_dataset))
test_size = len(drive_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(drive_dataset, [train_size, val_size, test_size])

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the model, loss function, and optimizer
model = FinalNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# =============================================================================
# import os
# import cv2
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from torchvision.transforms import ToTensor
# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import confusion_matrix
# 
# 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Define the dataset class
# class DRIVEDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = sorted(os.listdir(os.path.join(root_dir, 'images')))
#         self.mask_files = sorted(os.listdir(os.path.join(root_dir, 'masks')))
# 
#     def __len__(self):
#         return len(self.image_files)
# 
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, 'images', self.image_files[idx])
#         mask_name = os.path.join(self.root_dir, 'masks', self.mask_files[idx])
#         
#         image = cv2.imread(img_name)
#         mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
# 
#         if self.transform:
#             sample = {'image': image, 'mask': mask}
#             sample = self.transform(sample)
# 
#         return sample
# 
# # Convert images to PyTorch tensors
# class ToTensor(object):
#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#         # Convert image from BGR to RGB format and convert to tensor
#         image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
#         # Convert mask to tensor and resize from (H, W) to (1, H, W)
#         mask = torch.from_numpy(mask).unsqueeze(0).float()
#         return {'image': image, 'mask': mask}
# 
# #The directory path for the DRIVE dataset
# drive_dataset_dir = 'yeganeh/402/drive'
# 
# 
# # Load the dataset
# drive_dataset = DRIVEDataset(root_dir=drive_dataset_dir, transform=transform)
# 
# # Split the dataset into training and testing sets
# train_size = int(0.7 * len(drive_dataset))
# val_size = int(0.2 * len(drive_dataset))
# test_size = len(drive_dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = random_split(drive_dataset, [train_size, val_size, test_size])
# 
# # Define dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)
# test_loader = DataLoader(test_dataset, batch_size=32)
# 
# # Define the model architecture
# class FinalNetwork(nn.Module):
#     def __init__(self):
#         super(FinalNetwork, self).__init__()
#         # Define your model architecture here
# 
#     def forward(self, x):
#         # Define the forward pass of your model
#         return x
# 
# #Train
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     best_val_loss = float('inf')
#     early_stopping_counter = 0
# 
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in train_loader:
#             inputs, labels = batch['image'].to(device), batch['mask'].to(device)
# 
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             train_loss += loss.item() * inputs.size(0)
# 
#         train_loss /= len(train_loader.dataset)
# 
#         # Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs, labels = batch['image'].to(device), batch['mask'].to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)
# 
#         val_loss /= len(val_loader.dataset)
# 
#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stopping_counter = 0
#             torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
#         else:
#             early_stopping_counter += 1
#             if early_stopping_counter >= patience:
#                 print(f"No improvement in validation loss for {patience} epochs. Early stopping...")
#                 break
# 
#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
# 
# # Define the model, loss function, and optimizer
# model = FinalNetwork()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 
# # Train the model
# train_model(model, train_loader, val_loader, criterion, optimizer)
# 
# # Load the best model
# model.load_state_dict(torch.load('best_model.pt'))
# def specificity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)
#     return specificity
# 
# def sensitivity_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     sensitivity = tp / (tp + fn)
#     return sensitivity
# 
# def dice_score(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
#     return dice
# 
# def centerline_dice_score(y_true, y_pred):
#     # Implement the calculation of centerline-Dice score
#     pass
# 
# def matthews_correlation_coefficient(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return mcc
# 
# 
# # Evaluate the model on the test set
# model.eval()
# test_loss = 0.0
# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         inputs, labels = batch['image'].to(device), batch['mask'].to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item() * inputs.size(0)
#         predictions.append(outputs.cpu().numpy())
# 
# test_loss /= len(test_loader.dataset)
# 
# # Calculate evaluation metrics
# predictions = np.concatenate(predictions, axis=0)
# test_labels = np.concatenate([batch['mask'].numpy() for batch in test_loader], axis=0)
# 
# ACC = accuracy_score(test_labels.flatten(), (predictions > 0.5).flatten())
# SP = specificity_score(test_labels.flatten(), (predictions > 0.5).flatten())
# SE = sensitivity_score(test_labels.flatten(), (predictions > 0.5).flatten())
# Dice = dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
# clDice = centerline_dice_score(test_labels.flatten(), (predictions > 0.5).flatten())
# MCC = matthews_correlation_coefficient(test_labels.flatten(), (predictions > 0.5).flatten())
# 
# # Print evaluation results
# print("Test Loss:", test_loss)
# print("Matthews Correlation Coefficient:", MCC)
# print("Accuracy:", ACC)
# print("Specificity:", SP)
# print("Sensitivity:", SE)
# print("Dice Score:", Dice)
# print("Centerline Dice Score:", clDice)
# 
# =============================================================================
