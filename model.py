import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1_1 = nn.Linear(10, 1024)
        self.fc1_2 = nn.Linear(1024, 512)
        self.fc1_3 = nn.Linear(512, 64)

        self.fc2_1 = nn.Linear(10, 512)
        self.fc2_2 = nn.Linear(512, 64)

        self.fc3_1 = nn.Linear(128, 256)
        self.fc3_2 = nn.Linear(256, 64)
        self.fc3_3 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        # x1 : prediction
        # x2 : class
        x1 = self.fc1_1(x1)
        x1 = self.fc1_2(x1)
        x1 = self.fc1_3(x1)

        x2 = self.fc2_1(x2)
        x2 = self.fc2_2(x2)

        x = self.fc3_1(torch.cat((x1, x2), 1))
        x = self.fc3_2(x)
        x = torch.sigmoid(self.fc3_3(x))

        return x


class MIADataset(Dataset):
    def __init__(self, name, mia_data_dir):
        self.mia_data = pd.read_csv(os.path.join(mia_data_dir, name), header=None)

    def __len__(self):
        return len(self.mia_data)

    def __getitem__(self, idx):
        x1 = self.mia_data.iloc[idx, 0:-2].values
        x2 = self.mia_data.iloc[idx, -2]
        label = self.mia_data.iloc[idx, -1]

        return [x1, x2, label]
