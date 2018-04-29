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
parser.add_argument('--data-path', type=str, default='../data', metavar='PATH',
                    help='data path (default: ../data)')
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
    return len(p[1].split(' ')) < max_length and p[1].startswith(eng_prefixes) and len(p[1].split(' ')) >= 3

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
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count UNK, SOS and EOS
        
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

    print("Constructing training pairs...")
    max_ngrams_len = 0 
    len_train_pairs = len(train_pairs)

    for i in range(len_train_pairs):
        pair = train_pairs[i]
        output_lang.addSentence(pair[1])
        pair[0] =extract_ngrams(pair[1], order)
        if len(pair[0]) > max_ngrams_len:
            max_ngrams_len = len(pair[0])
        uwords = [t.text for t in nlp(str(pair[1]))]
        #print(uwords)
        if uwords[-1] in ('.', '!', '?'):
            del uwords[-1] # delete punctuation
        #print(uwords)
        random_word = random.choice(list(output_lang.index2word.values()))
        train_pairs.append([pair[0], [random_word], int(random_word in uwords)])
        pair[1] = [random.choice(uwords)]
        pair.append(1)

        uwords.remove(pair[1][0])
        if (len(uwords) > 0):
            train_pairs.append([pair[0], [random.choice(uwords)], 1])
        

    print("Constructing test pairs...")
    len_test_pairs = len(test_pairs)
    for i in range(len_test_pairs):
        pair = test_pairs[i]
        pair[0] = extract_ngrams(pair[1], order)
        if len(pair[0]) > max_ngrams_len:
            max_ngrams_len = len(pair[0])
        uwords = [t.text for t in nlp(str(pair[1]))]
        if uwords[-1] in ('.', '!', '?'):
            del uwords[-1] # delete punctuation
        random_word = random.choice(list(output_lang.index2word.values()))
        test_pairs.append([pair[0], [random_word], int(random_word in pair[1])])
        pair[1] = [random.choice(uwords)]
        pair.append(1)

        
    print("Max Ngrams length of all training and testing sentences:", max_ngrams_len)

    return input_lang, output_lang, train_pairs, test_pairs, max_ngrams_len

if __name__ == '__main__':
    args = parser.parse_args()
    
    input_lang, output_lang, train_pairs, test_pairs, max_ngrams_len = prepareData('eng', 'fra', 
        args.order, args.data_path, args.filter_pair, args.max_length, True)
    #vocab_ngrams = output_lang.createNGramDictionary()
    #lang = (output_lang.word2index, output_lang.word2count, output_lang.index2word, output_lang.n_words, 
    #    args.order, vocab_ngrams, max_ngrams_len)

    with open("train_pairs%d.pkl" % args.order, 'wb') as f:
        pkl.dump(train_pairs, f, protocol=pkl.HIGHEST_PROTOCOL) 
    with open("test_pairs%d.pkl" % args.order, 'wb') as f:
        pkl.dump(test_pairs, f, protocol=pkl.HIGHEST_PROTOCOL) 
    #with open("lang%d.pkl" % args.order, 'wb') as f:
    #    pkl.dump(lang, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # with open("lang.pkl", 'rb') as f:
    #     lang_load = pkl.load(f)
    # assert(lang_load == lang)
    
    print("Example training sentence pairs:")
    print(random.choice(train_pairs))
    print("Example testing sentence pairs:")
    print(random.choice(test_pairs))
