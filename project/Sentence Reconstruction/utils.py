from __future__ import unicode_literals, print_function, division

import spacy
import unicodedata
import string
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# nlp = spacy.load('en')
nlp = spacy.load("/scratch/zc807/nlu/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0")

# Define constants
UNK_token = 0
SOS_token = 1
EOS_token = 2

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

# Extract N-Grams
def extract_ngrams(sent, order):
    ngrams = []
    
    # tokenization
    uwords = [t.text for t in nlp(str(sent))]
    
    # extract ngrams
    for oo in range(1, order + 1):
        for ng in ([' '.join(t).strip() for t in zip(*[uwords[i:] for i in range(oo)])]):
            ngrams.append(ng)
            
    return ngrams

# Timeing and Plotting
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, args):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(args.data_path + "loss%d.jpg" % args.order)
    plt.close(fig)
