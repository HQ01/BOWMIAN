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
