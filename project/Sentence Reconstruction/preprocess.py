from __future__ import unicode_literals, print_function, division

import argparse
from io import open
from collections import OrderedDict
import random
import numpy as np
import pickle as pkl

from utils import *

###############################################
# Preprocessing settings
###############################################

parser = argparse.ArgumentParser(description='Data Preprocessing')
parser.add_argument('--data-path', type=str, default='.', metavar='PATH',
                    help='data path (default: current folder)')
parser.add_argument('--order', type=int, default=3, metavar='N',
                    help='order of ngram')
parser.add_argument('--no-filter-pair', dest='filter-pair', action='store_false',
                    help='disable pair filtering (default: enabled)')
parser.set_defaults(filter_pair=True)
parser.add_argument('--max-length', type=int, default=10, metavar='N',
                    help='maximum length of sentences, available only if pair filtering enabled (default: 10)')


###############################################
# Auxiliary functions for data preprocessing
###############################################

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p, max_length):
    return len(p[1].split(' ')) < max_length and p[1].startswith(eng_prefixes)

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


###############################################
# Core classes and functions
###############################################

class Lang:
    def __init__(self, name, order):
        self.name = name
        
        # for single words
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        
        # for ngrams
        self.order = order
        self.vocab0 = OrderedDict()

    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)
        ngrams = extract_ngrams(sent, self.order)
        for ng in ngrams:
            if ng in self.vocab0:
                self.vocab0[ng] += 1
            else:
                self.vocab0[ng] = 1
        return ngrams

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def createNGramDictionary(self):
        tokens = list(self.vocab0.keys())
        freqs = list(self.vocab0.values())
        sidx = np.argsort(freqs)[::-1]
        vocab = OrderedDict([(tokens[s], i) for i, s in enumerate(sidx)])
        return vocab

def readLangs(lang1, lang2, order, data_path, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    train_lines = open(data_path + '/train.txt', encoding='utf-8').\
        read().strip().split('\n')
    test_lines = open(data_path + '/test.txt', encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    train_pairs = [[normalizeString(s) for s in l.split('\t')] for l in train_lines]
    test_pairs = [[normalizeString(s) for s in l.split('\t')] for l in test_lines]

    # Reverse pairs, make Lang instances
    if reverse:
        train_pairs = [list(reversed(p)) for p in train_pairs]
        test_pairs = [list(reversed(p)) for p in test_pairs]
        input_lang = Lang(lang2, order)
        output_lang = Lang(lang1, order)
    else:
        input_lang = Lang(lang1, order)
        output_lang = Lang(lang2, order)

    return input_lang, output_lang, train_pairs, test_pairs

def prepareData(lang1, lang2, order, data_path, filter_pair, max_length, reverse=False):
    input_lang, output_lang, train_pairs, test_pairs = readLangs(lang1, lang2, order, data_path, reverse)
    print("Read %s training sentence pairs" % len(train_pairs))
    print("Read %s testing sentence pairs" % len(test_pairs))

    if filter_pair:
        train_pairs = filterPairs(train_pairs, max_length)
        test_pairs = filterPairs(test_pairs, max_length)
    print("Trimmed to %s training sentence pairs" % len(train_pairs))
    print("Trimmed to %s testing sentence pairs" % len(test_pairs))

    print("Counting words and constructing training pairs...")
    max_ngrams_len = 0 
    for pair in train_pairs:
        #input_lang.addSentence(pair[0])
        pair[0] = output_lang.addSentence(pair[1])
        if len(pair[0]) > max_ngrams_len:
            max_ngrams_len = len(pair[0])
    print("Counted words in training sentences:")
#     print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("Constructing test pairs...")
    for pair in test_pairs:
        #input_lang.addSentence(pair[0])
        pair[0] = extract_ngrams(pair[1], order)
        if len(pair[0]) > max_ngrams_len:
            max_ngrams_len = len(pair[0])
    print("Max Ngrams length of all training and testing sentences:", max_ngrams_len)

    return input_lang, output_lang, train_pairs, test_pairs, max_ngrams_len

if __name__ == '__main__':
    args = parser.parse_args()
    
    input_lang, output_lang, train_pairs, test_pairs, max_ngrams_len = prepareData('eng', 'fra', 
        args.order, args.data_path, args.filter_pair, args.max_length, True)
    vocab_ngrams = output_lang.createNGramDictionary()
    lang = (output_lang.word2index, output_lang.word2count, output_lang.index2word, output_lang.n_words, 
        args.order, vocab_ngrams, max_ngrams_len)

    with open("pairs.pkl", 'wb') as f:
        pkl.dump((train_pairs, test_pairs), f, protocol=pkl.HIGHEST_PROTOCOL) 
    with open("lang.pkl", 'wb') as f:
        pkl.dump(lang, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # with open("lang.pkl", 'rb') as f:
    #     lang_load = pkl.load(f)
    # assert(lang_load == lang)
    
    print("Example training sentence pairs:")
    print(random.choice(train_pairs))
    print("Example testing sentence pairs:")
    print(random.choice(test_pairs))
