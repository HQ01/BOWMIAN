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
parser.add_argument('--order', type=int, default='3', metavar='N',
                    help='order of ngram')
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
        self.order = lang_load[4]
        self.vocab_ngrams = lang_load[5]
        self.max_ngrams_len = lang_load[6]

def indexesFromSentence(lang, sentence):
    result = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            result.append(lang.word2index[word])
        else:
            result.append(UNK_token)
    return result

def variableFromSentence(lang, sentence, args):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if args.cuda:
        return result.cuda()
    else:
        return result

def indexesFromNGramList(vocab, ngram_list, num_words, args):
    result = [[] * args.order]
    for ng in ngram_list:
        slot = len(ng.split(' ')) - 1
        if ng in vocab:
            idx = vocab[ng]
            if idx > num_words:
                result[slot].append(UNK_token)
            else:
                result[slot].append(idx)
        else:
            result[slot].append(UNK_token)
    return result

def variableFromNGramList(vocab, ngram_list, num_words, args):
    indexes = indexesFromNGramList(vocab, ngram_list, num_words, args)
    result = Variable(torch.LongTensor(indexes))
    if args.cuda:
        return result.cuda()
    else:
        return result
    
def variablesFromPair(pair, lang, args):
    input_variable = variableFromNGramList(lang.vocab_ngrams, pair[0], args.num_words, args)
    target_variable = variableFromSentence(lang, pair[1], args)
    return (input_variable, target_variable)


###############################################
# Training
###############################################

teacher_forcing_ratio = 0.5

def train_attn(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, args):
    use_cuda = args.cuda
    max_length = args.max_length

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    # encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if use_cuda:
        decoder_input = decoder_input.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if use_cuda: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break

    # Backpropagation
    loss.backward()
    nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
    nn.utils.clip_grad_norm(decoder.parameters(), args.clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length

def trainEpochs(encoder, decoder, lang, pairs, args):
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
        training_pairs = [variablesFromPair(pair, lang, args)
                      for pair in pairs]

        for training_pair in training_pairs:
            iter += 1
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train_attn(input_variable, target_variable, encoder,
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

def evaluate(encoder, decoder, sentence, lang, args):
    use_cuda = args.cuda
    max_length = args.max_length

    input_variable = variable_from_sentence(lang, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluateRandomly(encoder, decoder, pairs, lang, args, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], lang, args)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateTestingPairs(encoder, decoder, pairs, lang, args):
    list_cand = []
    list_ref = []
    print("Evaluating {} testing sentences...".format(len(pairs)))
    for pair in pairs:
        output_words, attentions = evaluate(encoder, decoder, pair[0], lang, args)
        output_sentence = ' '.join(output_words)
        list_cand.append(output_sentence)
        list_ref.append(pair[1])
    print("{} score: {}".format(args.metric, score(list_cand, list_ref, args.order, args.metric)))

def evaluate_and_show_attention(encoder, decoder, input_sentence, lang, args):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, lang, args)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

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

    # Set encoder and decoder
    encoder = NGramEncoder(args.num_words, args.hidden_size, args.mode)
    decoder = BahdanauAttnDecoderRNN(args.hidden_size, lang.n_words, args.cuda)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # Train and evalute
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, args)
    trainEpochs(encoder, decoder, lang, train_pairs, args)
    print("Evaluate randomly on training sentences:")
    evaluateRandomly(encoder, decoder, train_pairs, lang, args)
    print("Evaluate randomly on testing sentences:")
    evaluateRandomly(encoder, decoder, test_pairs, lang, args)
    evaluateTestingPairs(encoder, decoder, test_pairs, lang, args)

    # Export trained embedding weights
    if args.cuda:
        embedding_weights = encoder.embeddingBag.weight.data.cpu().numpy()
    else: 
        embedding_weights = encoder.embeddingBag.weight.data.numpy()
    with open("embedding_weights%d.pkl" % args.order, 'wb') as f:
        pkl.dump(embedding_weights, f, protocol=pkl.HIGHEST_PROTOCOL)
