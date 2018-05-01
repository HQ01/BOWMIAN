from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class NGramEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, mode):
        super(NGramEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embeddingBag = nn.EmbeddingBag(input_size, hidden_size, mode=mode)
    
    def forward(self, input):
        return self.embeddingBag(input.view(1, -1)).view(1, 1, -1) # convert to 2-D tensor first for embeddingBag

class MLP(nn.Module):
    def __init__(self, input_size, class_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_size)
        
    def forward(self, embedding):
        out = embedding.view(1, -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
