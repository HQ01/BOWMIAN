from __future__ import unicode_literals, print_function, division

import argparse
from io import open
from collections import OrderedDict
import unicodedata
import string
import re
import random
import numpy as np
import pickle as pkl
import spacy

nlp = spacy.load('en')

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

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Extract ngrams and build up a dictionary
def extract_ngrams(vocab, sent, order):
    ngrams = []
    
    # tokenization
    uwords = [t.text for t in nlp(str(sent))]
    
    # extract ngrams
    for oo in range(1, order + 1):
        for ng in set([' '.join(t).strip() for t in zip(*[uwords[i:] for i in range(oo)])]):
            ngrams.append(ng)
            if ng in vocab:
                vocab[ng] += 1
            else:
                vocab[ng] = 1

    return vocab, ngrams


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

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        self.vocab0, ngrams = extract_ngrams(self.vocab0, sentence, self.order)
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
    lines = open(data_path + '/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, order)
        output_lang = Lang(lang1, order)
    else:
        input_lang = Lang(lang1, order)
        output_lang = Lang(lang2, order)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, order, data_path, filter_pair, max_length, reverse=False):
    data = []
    input_lang, output_lang, pairs = readLangs(lang1, lang2, order, data_path, reverse)
    print("Read %s sentence pairs" % len(pairs))
    if filter_pair:
        pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #input_lang.addSentence(pair[0])
        pair[0] = output_lang.addSentence(pair[1])

    print("Counted words:")
#     print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

if __name__ == '__main__':
    args = parser.parse_args()
    
    input_lang, output_lang, pairs = prepareData('eng', 'fra', 
        args.order, args.data_path, args.filter_pair, args.max_length, True)
    vocab_ngrams = output_lang.createNGramDictionary()
    
    with open("lang_word2index.pkl", 'wb') as f:
        pkl.dump(output_lang.word2index, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open("lang_index2word.pkl", 'wb') as f:
        pkl.dump(output_lang.index2word, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open("dict_ngram.pkl", 'wb') as f:
        pkl.dump(vocab_ngrams, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # with open("lang_word2index.pkl", 'rb') as f:
    #     lang_word2index_load = pkl.load(f)
    # with open("lang_index2word.pkl", 'rb') as f:
    #     lang_index2word_load = pkl.load(f)
    # with open("dict_ngram.pkl", 'rb') as f:
    #     vocab_ngrams_load = pkl.load(f)
    # assert(lang_word2index_load == output_lang.word2index)
    # assert(lang_index2word_load == output_lang.index2word)
    # assert(vocab_ngrams == vocab_ngrams_load)
    
    print(random.choice(pairs))
