# -*- coding: utf-8 -*-


import torch
from sklearn.metrics import confusion_matrix

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

def dice_score(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (torch.sum(y_true) + torch.sum(y_pred))
    return dice

def centerline_dice_score(y_true, y_pred):
    # Implement the calculation of centerline-Dice score
    pass

def matthews_correlation_coefficient(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc
