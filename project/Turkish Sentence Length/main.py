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

parser = argparse.ArgumentParser(description='Infering Sentence Length with NGrams + MLP')
parser.add_argument('--order', type=int, default='3', metavar='N',
                    help='order of ngram')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data-path', type=str, default='/scratch/zc807/nlu/tur_sentence_length', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: /scratch/zc807/nlu/tur_sentence_length)')
parser.add_argument('--load-data-path', type=str, default='/scratch/zc807/nlu/tur_embedding_weights', metavar='PATH',
                    help='data path to load embedding weights (default: /scratch/zc807/nlu/tur_embedding_weights)')
parser.add_argument('--mode', type=str, choices=['sum', 'mean'], default='sum', metavar='MODE',
                    help='mode of bag-of-n-gram representation (default: sum)')
parser.add_argument('--num-words', type=int, default='50000', metavar='N',
                    help='maximum ngrams vocabulary size to use (default: 50000')
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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.set_defaults(max_length=100)


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
                # result.append(UNK_token)
                pass
            else:
                result.append(idx)
        else:
            # result.append(UNK_token)
            pass
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
    # bins: (1-2), (3-4), (5-6), (7-8), (9-10), (11-12), (13-14), (15-16), (17-18), (18+)
    category = min(math.floor((length - 1) / 2), 9)
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

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
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
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % args.plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print("Epoch {}/{} finished".format(epoch, args.n_epochs))

    showPlot(plot_losses, args)


###############################################
# Evaluation
###############################################

def evaluateRandomly(encoder, net, pairs, lang, args, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('ref sent:', pair[2])
        print('=', pair[1])

        input_variable = variableFromNGramList(lang.vocab_ngrams, pair[0], args.num_words, args)
        input_length = input_variable.size()[0]

        encoder_hidden = encoder(input_variable)

        outputs = net(encoder_hidden)
        predict = torch.max(outputs, 1)[1].data[0]

        if predict < 9:
            print('< {}-{}'.format(predict * 2 + 1, predict * 2 + 2))
        else:
            print('< 18+')
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
        label = min(math.floor((int(pair[1]) - 1) / 2), 9)
        
        total += 1
        if (predict == label):
            correct += 1

    print('Accuracy of the network on the sentence length test set: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.hpc:
        args.data_path = '.'
        args.load_data_path = '../embedding_weights'

    # Print settings
    print("hpc mode: {}".format(args.hpc))
    print("order: {}".format(args.order))
    print("mode: {}".format(args.mode))
    print("max ngram dictionary size: {}".format(args.num_words))
    print("hidden-size: {}".format(args.hidden_size))
    print("n-epochs: {}".format(args.n_epochs))
    print("print-every: {}".format(args.print_every))
    print("plot-every: {}".format(args.plot_every))
    print("lr: {}".format(args.lr))
    print("use cuda: {}".format(args.cuda))

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/pairs%d.pkl" % args.order, 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/lang%d.pkl" % args.order, 'rb') as f:
        lang_load = pkl.load(f)
    lang = Lang(lang_load)
    args.max_length = lang.max_ngrams_len

    # Set encoder and net
    encoder = NGramEncoder(args.num_words, args.hidden_size, args.mode)
    with open(args.load_data_path + "/embedding_weights%d.pkl" % args.order, 'rb') as f:
        embedding_weights = pkl.load(f)
        encoder.embeddingBag.weight.data.copy_(torch.from_numpy(embedding_weights))
    net = MLP(args.hidden_size, class_size=10)
    if args.cuda:
        encoder = encoder.cuda()
        net = net.cuda()

    # Train and evalute
    print("Start")
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, net, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, net, test_pairs, lang, args)
    evaluateTestingPairs(encoder, net, test_pairs, lang, args)
    trainEpochs(encoder, net, lang, train_pairs, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, net, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, net, test_pairs, lang, args)
    evaluateTestingPairs(encoder, net, test_pairs, lang, args)
    print("Finished\n")
