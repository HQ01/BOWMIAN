from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, class_size=7):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_size)
        
    def forward(self, embedding):
        out = embedding.view(1, -1)
        #print(out)
        out = nn.Tanh(self.fc1(out))
        #print(out)
        out = nn.Tanh(self.fc2(out))
        out = self.fc3(out)
        return out

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

class NGramEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, mode):
        super(NGramEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embeddingBag = nn.EmbeddingBag(input_size, hidden_size, mode=mode)
    
    def forward(self, input):
        return self.embeddingBag(input.view(1, -1)).view(1, 1, -1) # convert to 2-D tensor first for embeddingBag

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, use_cuda):
        super(DecoderRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
