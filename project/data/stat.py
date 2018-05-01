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
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data-path', type=str, default='/scratch/zc807/nlu/data', metavar='PATH',
                    help='data path (default: /scratch/zc807/nlu/data)')
parser.add_argument('--save-data-path', type=str, default='/scratch/zc807/nlu/sentence_reconstruction', metavar='PATH',
                    help='data path to save pairs.pkl and lang.pkl (default: /scratch/zc807/nlu/sentence_reconstruction)')
parser.add_argument('--order', type=int, default=3, metavar='N',
                    help='order of ngram')
parser.add_argument('--num-pairs', type=int, default=20000, metavar='N',
                    help='number of training pairs to use, 4 times of that of testing pairs')


###############################################
# Core classes and functions
###############################################

class Lang:
    def __init__(self, name, order):
        self.name = name
        
        # for single words
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count UNK, SOS and EOS

    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

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

def prepareData(lang1, lang2, order, data_path, num_pairs, reverse=False):
    input_lang, output_lang, train_pairs, test_pairs = readLangs(lang1, lang2, order, data_path, reverse)
    print("Read %s training sentence pairs" % len(train_pairs))
    print("Read %s testing sentence pairs" % len(test_pairs))

    assert(len(train_pairs) > num_pairs)
    assert(len(test_pairs) > int(num_pairs/4))
    train_pairs = train_pairs[:num_pairs]
    test_pairs = test_pairs[:int(num_pairs/4)]
    print("Trimmed to %s training sentence pairs" % len(train_pairs))
    print("Trimmed to %s testing sentence pairs" % len(test_pairs))

    cnt_short_sent_train = 0
    cnt_long_sent_train = 0
    print("Counting words and constructing training pairs...")
    for pair in train_pairs:
        pair[1] = pair[0]
        sent_length = len([t.text for t in nlp(str(pair[1]))])
        if sent_length > (6 + 1): # extra 1 for ending punctuation
            cnt_long_sent_train += 1
        else:
            cnt_short_sent_train += 1
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words in training sentences:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("Number of short sentences:", cnt_short_sent_train)
    print("Number of long sentences:", cnt_long_sent_train)

    cnt_short_sent_test = 0
    cnt_long_sent_test = 0 
    print("Constructing testing pairs...")
    for pair in test_pairs:
        pair[1] = pair[0]
        sent_length = len([t.text for t in nlp(str(pair[1]))])
        if sent_length > (6 + 1): # extra 1 for ending punctuation
            cnt_long_sent_test += 1
        else:
            cnt_short_sent_test += 1
    print("Number of short sentences:", cnt_short_sent_test)
    print("Number of long sentences:", cnt_long_sent_test)

    return input_lang, output_lang, train_pairs, test_pairs

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.hpc:
        args.data_path = '.'
        args.save_data_path = '.'

    print("hpc mode: {}".format(args.hpc))
    print("num-pairs: {}".format(args.num_pairs))
    input_lang, output_lang, train_pairs, test_pairs = prepareData('eng', 'eng', 
        args.order, args.data_path, args.num_pairs, False)
    input_lang = (input_lang.word2index, input_lang.word2count, input_lang.index2word, input_lang.n_words)
    output_lang = (output_lang.word2index, output_lang.word2count, output_lang.index2word, output_lang.n_words)
    
    print("Example training sentence pairs:")
    print(random.choice(train_pairs))
    print("Example testing sentence pairs:")
    print(random.choice(test_pairs))
    