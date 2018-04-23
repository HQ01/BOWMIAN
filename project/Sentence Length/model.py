from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, class_size=6, act='Tanh'):
        super(MLPNet,self).__init__()
        
        self.hids = []
        
        self.fc1 = nn.Linear(100, 64)
        self.tanh = eval('nn.{}'.format(act))()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_size)
        
        self.hid_modules = nn.ModuleList([h[0] for h in self.hids])

        #self.classifier = nn.Linear(options['n_hid'], 2)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, sentEmb):

        h = sentEmb
        h = self.fc1(h)
        h = self.tanh(h)
        h = self.fc2(h)
        h = self.tanh(h)
        h = self.fc3(h)
        
        #z = self.classifier(h)
        m = nn.Softmax(dim=1)
        z = m(h)

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
