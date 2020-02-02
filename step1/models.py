import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EmbeddedVectorModel(nn.Module):

    def __init__(self, output_dim):
        super(EmbeddedVectorModel, self).__init__()
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

