from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda):
        super(EncoderRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

class MLP_wc(nn.Module):
    def __init__(self,input_size,class_size = 2):
        super(MLP_wc, self).__init__()
        self.fc1 = nn.Linear(input_size*2,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_size)


    def forward(self,embedding):
        out = embedding.view(1,-1)
        #aux = embedding.view(1,-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        #print("before softmax: ", out)
        m = nn.Softmax(dim=1)
        z = m(out)
        #print("after softmax: ", z)
        return z
