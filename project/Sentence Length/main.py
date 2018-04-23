from __future__ import unicode_literals, print_function, division

import argparse
import time
from io import open
import random
import math
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils import *
from model import *
from metric import score

###############################################
# Training settings
###############################################

parser = argparse.ArgumentParser(description='Sentence Reconstruction with NGrams')
parser.add_argument('--data-path', type=str, default='.', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: current folder)')
parser.add_argument('--mode', type=str, choices=['sum', 'mean'], default='sum', metavar='MODE',
                    help='mode of bag-of-n-gram representation (default: sum)')
parser.add_argument('--metric', type=str, default='ROUGE', metavar='METRIC',
                    help='metric to use (default: ROUGE; BLEU and BLEU_clip available)')
parser.add_argument('--num-words', type=int, default='10000', metavar='N',
                    help='maximum ngrams vocabulary size to use (default: 10000')
parser.add_argument('--hidden-size', type=int, default='256', metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--n-epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--n-iters', type=int, default=3000, metavar='N',
                    help='number of iters to train (default: 3000), for testing only')
parser.add_argument('--print-every', type=int, default='1000', metavar='N',
                    help='print every (default: 1000) iters')
parser.add_argument('--plot-every', type=int, default='100', metavar='N',
                    help='plot every (default: 100) iters')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--order', type=int, default='2', metavar='N',
                    help='order of ngram (set by preprocessing)')
parser.add_argument('--max-length', type=int, default='100', metavar='N',
                    help='max-ngrams-length (set by preprocessing)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


###############################################
# Preparing training data
###############################################

class Lang:
    def __init__(self, lang_load):
        self.word2index = lang_load[0]
        self.word2count = lang_load[1]
        self.index2word = lang_load[2]
        self.n_words = lang_load[3]
        self.order = lang_load[4]
        self.vocab_ngrams = lang_load[5]
        self.max_ngrams_len = lang_load[6]

def indexesFromNGramList(vocab, ngram_list, num_words):
    result = []
    for ng in ngram_list:
        if ng in vocab:
            idx = vocab[ng]
            if idx > num_words:
                result.append(UNK_token)
            else:
                result.append(idx)
        else:
            result.append(UNK_token)
    return result

def variableFromNGramList(vocab, ngram_list, num_words, args):
    indexes = indexesFromNGramList(vocab, ngram_list, num_words)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if args.cuda:
        return result.cuda()
    else:
        return result

def variableFromSentenceLength(label, args):
    length = int(label)
    # bins: (1-4), (5-8), (9-12), (13-16), (17-20), (20-24), (24+)
    category = min(math.floor((length - 1) / 4), 6)
    result = Variable(torch.LongTensor([category]))
    if args.cuda:
        return result.cuda()
    else:
        return result
    
def variablesFromPair(pair, lang, args):
    input_variable = variableFromNGramList(lang.vocab_ngrams, pair[0], args.num_words, args)
    target_variable = variableFromSentenceLength(pair[1], args)
    return (input_variable, target_variable)


###############################################
# Training
###############################################

def train(input_variable, target_variable, encoder, decoder, net_optimizer, criterion, args):
    use_cuda = args.cuda
    max_length = args.max_length

    loss = 0

    # encoder_optimizer.zero_grad()
    net_optimizer.zero_grad()

    encoder_hidden = encoder(input_variable)
    net_output = net(encoder_hidden)
    loss = criterion(net_output, target_variable)
    loss.backward()

    # encoder_optimizer.step()
    net_optimizer.step()

    return loss.data[0]

def trainEpochs(encoder, net, lang, pairs, args):
    n_epochs = args.n_epochs
    print_every = args.print_every
    plot_every = args.plot_every
    learning_rate = args.lr

    start = time.time()
    iter = 0
    n_iters = n_epochs * len(pairs)
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    net_optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.n_epochs + 1):
        random.shuffle(pairs)
        training_pairs = [variablesFromPair(pair, lang, args)
                      for pair in pairs]

        for training_pair in training_pairs:
            iter += 1
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder,
                    net, net_optimizer, criterion, args)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % args.print_every == 0:
                evaluateRandomly(encoder, net, pairs, lang, args)
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % args.plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print("Epoch {}/{} finished".format(epoch, args.n_epochs))

    showPlot(plot_losses)


###############################################
# Evaluation
###############################################

def evaluateRandomly(encoder, net, pairs, lang, args, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])

        input_variable = variableFromNGramList(lang.vocab_ngrams, pair[0], args.num_words, args)
        input_length = input_variable.size()[0]

        encoder_hidden = encoder(input_variable)

        outputs = net(encoder_hidden)
        predict = torch.max(outputs, 1)[1].data[0]

        if predict < 6:
            print('< {}-{}'.format(predict * 4 + 1, predict * 4 + 4))
        else:
            print('< 24+')
        print('')

def evaluateTestingPairs(encoder, net, pairs, lang, args):
    use_cuda = args.cuda
    max_length = args.max_length

    correct = 0
    total = 0
    for pair in pairs:
        input_variable = variableFromNGramList(lang.vocab_ngrams, pair[0], args.num_words, args)
        input_length = input_variable.size()[0]

        encoder_hidden = encoder(input_variable)

        outputs = net(encoder_hidden)
        predict = torch.max(outputs, 1)[1].data[0]
        label = min(math.floor((int(pair[1]) - 1) / 4), 6)
        
        total += 1
        if (predict == label):
            correct += 1

    print('Accuracy of the network on the sentence length test set: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/pairs.pkl", 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/lang.pkl", 'rb') as f:
        lang_load = pkl.load(f)
    lang = Lang(lang_load)
    args.order = lang.order
    args.max_length = lang.max_ngrams_len

    # Set encoder and net
    net = MLP(args.hidden_size, class_size=7)
    encoder = NGramEncoder(args.num_words, args.hidden_size, args.mode)
    if args.cuda:
        encoder = encoder.cuda()
        net = net.cuda()

    # Load pretrained embedding weights
    with open("embedding_weights.pkl", 'rb') as f:
        embedding_weights = pkl.load(f)
        encoder.embeddingBag.weight.data.copy_(torch.from_numpy(embedding_weights))

    # Train and evalute
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, net, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, net, test_pairs, lang, args)
    trainEpochs(encoder, net, lang, train_pairs, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, net, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, net, test_pairs, lang, args)
    evaluateTestingPairs(encoder, net, test_pairs, lang, args)
