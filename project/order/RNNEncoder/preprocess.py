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
parser.add_argument('--save-data-path', type=str, default='/scratch/zc807/nlu/word_order', metavar='PATH',
                    help='data path to save pairs.pkl and lang.pkl (default: /scratch/zc807/nlu/word_order)')
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
    max_ngrams_len = 0

    cleaned_train_pairs = []
    cleaned_test_pairs = []
    print("Constructing training pairs...")
    for i in range(len(train_pairs)):
        pair = train_pairs[i]
        output_lang.addSentence(pair[0])
        pair[0] = pair[1]
        uwords = [t.text for t in nlp(str(pair[1]))]
        #print(uwords)
        if uwords[-1] in ('.', '!', '?'):
            del uwords[-1] # delete punctuation

        # sentence too short!
        if len(uwords) < 2:
            continue

        word1 = random.choice(uwords)
        word1_ind = uwords.index(word1)
        if (word1_ind == len(uwords) - 1):
            word2 = random.choice(uwords[:word1_ind])
            temp = word2
            word2 = word1
            word1 = temp
        else:
            word2 = random.choice(uwords[word1_ind+1:])

        word3 = random.choice(uwords)
        word3_ind = uwords.index(word3)
        if (word3_ind == len(uwords) - 1):
            word4 = random.choice(uwords[:word3_ind])
            temp = word4
            word4 = word3
            word3 = temp
        else:
            word4 = random.choice(uwords[word3_ind+1:])

        pair[1] = [word1, word2]
        pair.append(1)

        cleaned_train_pairs.append(pair)
        cleaned_train_pairs.append([pair[0], [word4, word3], 0])

    print("Constructing test pairs...")
    for i in range(len(test_pairs)):
        pair = test_pairs[i]
        pair[0] = pair[1]
        uwords = [t.text for t in nlp(str(pair[1]))]
        #print(uwords)
        if uwords[-1] in ('.', '!', '?'):
            del uwords[-1] # delete punctuation
        
        # sentence too short!
        if len(uwords) < 2:
            continue

        word1 = random.choice(uwords)
        word1_ind = uwords.index(word1)
        word2 = random.choice(uwords[word1_ind::])

        word3 = random.choice(uwords)
        word3_ind = uwords.index(word3)
        word4 = random.choice(uwords[word3_ind::])

        pair[1] = [word1, word2]
        pair.append(1)

        cleaned_test_pairs.append(pair)
        cleaned_test_pairs.append([pair[0], [word4, word3], 0])

    return input_lang, output_lang, cleaned_train_pairs, cleaned_test_pairs, max_ngrams_len

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.hpc:
        args.data_path = '../../data'
        args.save_data_path = '.'
    
    print("hpc mode: {}".format(args.hpc))
    print("order: {}".format(args.order))
    print("num-sentence-pairs: {}".format(args.num_pairs))

    input_lang, output_lang, train_pairs, test_pairs, max_ngrams_len = prepareData('eng', 'fra', 
        args.order, args.data_path, args.num_pairs, True)
    lang = (output_lang.word2index, output_lang.word2count, output_lang.index2word, output_lang.n_words)

    print("num-cleaned-training-pairs (2 pairs per cleaned sentence): {}".format(len(train_pairs)))
    print("num-cleaned-testing-pairs (2 pairs per cleaned sentence): {}".format(len(test_pairs)))

    with open(args.save_data_path + "/RNNEncoder_pairs.pkl", 'wb') as f:
        pkl.dump((train_pairs, test_pairs), f, protocol=pkl.HIGHEST_PROTOCOL) 
    with open(args.save_data_path + "/RNNEncoder_lang.pkl", 'wb') as f:
        pkl.dump(lang, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    print("Example training sentence pairs:")
    print(random.choice(train_pairs))
    print("Example testing sentence pairs:")
    print(random.choice(test_pairs))
    print("Finished\n")
