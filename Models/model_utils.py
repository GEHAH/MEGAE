import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import shutil
import numpy as np
import dgl
import random
import json
import csv
import pickle
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, random_split

#损失函数
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds):
        image_embeds = F.normalize(image_embeds, dim=-1, p=2)
        text_embeds = F.normalize(text_embeds, dim=-1, p=2)

        similarity_matrix = F.cosine_similarity(image_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=-1) / self.temperature
        positives = torch.diag(similarity_matrix)
        nce_loss = -torch.log(torch.exp(positives) / torch.exp(similarity_matrix).sum(dim=-1)).mean()

        return nce_loss
    

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass
def evaluat_metrics(output, label):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    pre_y = (torch.sigmoid(output) > 0.5).numpy()
    truth_y = label.numpy()
    N, C = pre_y.shape

    for i in range(N):
        for j in range(C):
            if pre_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    TP += 1
                else:
                    TN += 1
            elif truth_y[i][j] == 1:
                FN += 1
            elif truth_y[i][j] == 0:
                FP += 1

        sum = N*C
        Accuracy = (TP + TN) / (sum + 1e-10)
        Precision = TP / (TP + FP + 1e-10)
        Recall = TP / (TP + FN + 1e-10)
        F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)
        # fpr, tpr, thresholds = metrics.roc_curve(pre_y, truth_y, pos_label=1)
        # AUC = metrics.auc(fpr, tpr)
    aupr_entry_1 = pre_y
    aupr_entry_2 = truth_y
    aupr = np.zeros(7)
    for i in range(7):
        precision, recall, _ = metrics.precision_recall_curve(aupr_entry_1[:, i], aupr_entry_2[:, i])
        aupr[i] = metrics.auc(recall, precision)
    AUPR = aupr.mean()

    return F1_score,AUPR



