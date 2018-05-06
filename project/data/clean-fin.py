from __future__ import unicode_literals, print_function, division

import argparse
import unicodedata
import string
import re
from io import open

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

dictionary = dict()
lines = open('eng-fin.txt', encoding='utf-8').\
        read().strip().split('\n')
pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
with open("eng-fin_cleaned.txt", 'w') as f:
    for l in lines:
        pair = [normalizeString(s) for s in l.split('\t')]
        if pair[0] in dictionary:
            pass
        else:
            f.write(l + '\n')
            dictionary[pair[0]] = 1
