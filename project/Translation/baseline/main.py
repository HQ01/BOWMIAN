from __future__ import unicode_literals, print_function, division

import argparse
import time
from io import open
import random
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
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data-path', type=str, default='/scratch/zc807/nlu/translation', metavar='PATH',
                    help='data path of pairs.pkl and lang.pkl (default: /scratch/zc807/nlu/translation)')
parser.add_argument('--model', type=str, default='', metavar='MODEL',
                    help='model architecture (default: )')
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
parser.add_argument('--clip', type=float, default=10, metavar='CLIP',
                    help='gradient clip threshold (default: 10)')
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

def indexesFromSentence(lang, sentence):
    result = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            result.append(lang.word2index[word])
        else:
            result.append(UNK_token)
    return result

def variableFromSentence(lang, sentence, args):
    use_cuda = args.cuda
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, input_lang, output_lang, args):
    input_variable = variableFromSentence(input_lang, pair[0], args)
    target_variable = variableFromSentence(output_lang, pair[1], args)
    return (input_variable, target_variable)


###############################################
# Training
###############################################

teacher_forcing_ratio = 0.5
 
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args):
    use_cuda = args.cuda
    max_length = args.max_length

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    # Clip gradient
    nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
    nn.utils.clip_grad_norm(decoder.parameters(), args.clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainEpochs(encoder, decoder, input_lang, output_lang, pairs, args):
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
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, args.n_epochs + 1):
        random.shuffle(pairs)
        training_pairs = [variablesFromPair(pair, input_lang, output_lang, args)
                      for pair in pairs]

        for training_pair in training_pairs:
            iter += 1
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, args)
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

    showPlot(plot_losses)

###############################################
# Evaluation
###############################################

def evaluate(encoder, decoder, sentence, input_lang, output_lang, args):
    use_cuda = args.cuda
    max_length = args.max_length

    input_variable = variableFromSentence(input_lang, sentence, args)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
#         decoder_output, decoder_hidden, decoder_attention = decoder(
#             decoder_input, decoder_hidden, encoder_outputs)
#         decoder_attentions[di] = decoder_attention.data
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

#     return decoded_words, decoder_attentions[:di + 1]
    return decoded_words

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, args, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, args)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateTestingPairs(encoder, decoder, pairs, input_lang, output_lang, args):
    list_cand = []
    list_ref = []
    print("Evaluating {} testing sentences...".format(len(pairs)))
    for pair in pairs:
        output_words = evaluate(encoder, decoder, pair[0], input_lang, output_lang, args)
        output_sentence = ' '.join(output_words)
        list_cand.append(output_sentence)
        list_ref.append(pair[1])
    print("{} score: {}".format(args.metric, score(list_cand, list_ref, args.order, args.metric)))

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.hpc:
        args.data_path = '.'

    # Print settings
    print("metric: {}".format(args.metric))
    print("hidden-size: {}".format(args.hidden_size))
    print("n-epochs: {}".format(args.n_epochs))
    print("print-every: {}".format(args.print_every))
    print("plot-every: {}".format(args.plot_every))
    print("lr: {}".format(args.lr))
    print("clip: {}".format(args.clip))

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load pairs.pkl and lang.pkl
    with open(args.data_path + "/baseline_pairs.pkl", 'rb') as f:
        (train_pairs, test_pairs) = pkl.load(f)
    with open(args.data_path + "/baseline_lang.pkl", 'rb') as f:
        (input_lang_load, output_lang_load) = pkl.load(f)
    input_lang = Lang(input_lang_load)
    output_lang = Lang(output_lang_load)
    args.max_length = 50

    # Set encoder and decoder
    encoder = EncoderRNN(args.num_words, args.hidden_size, args.cuda)
    decoder = DecoderRNN(args.hidden_size, output_lang.n_words, args.cuda)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Train and evalute
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, input_lang, output_lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, input_lang, output_lang, args)
    trainEpochs(encoder, decoder, input_lang, output_lang, train_pairs, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, input_lang, output_lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, input_lang, output_lang, args)
    evaluateTestingPairs(encoder, decoder, test_pairs, input_lang, output_lang, args)
    print("Finished")
