import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Subtask1Model(nn.Module):

    def __init__(self, output_dim):
        super(Subtask1Model, self).__init__()
        # network hyper parameters
        self.dropout_factor = 0.5

        # network architecture
        self.input = nn.Linear(2669, 256)
        self.dense = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_factor)
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dropout_factor)
        x = torch.sigmoid(self.dense(x)) # UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
        return x
    
    def get_activation(self, x):
        return F.relu(self.input(x))

class Subtask2Model(nn.Module):

    def __init__(self, output_dim):
        super(Subtask2Model, self).__init__()
        # network hyper parameters
        self.dropout_factor = 0.5

        # network architecture
        self.input = nn.Linear(1024, output_dim)

    def forward(self, meta):
        
        x = torch.cat((
            F.normalize(meta[0:256], p=2, dim=0),
            F.normalize(meta[256:512], p=2, dim=0),
            F.normalize(meta[512:768], p=2, dim=0),
            F.normalize(meta[768:1024], p=2, dim=0)),0)
        x = F.dropout(x, p=self.dropout_factor)
        x = torch.sigmoid(self.input(x)) # UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
        return x

