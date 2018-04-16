from __future__ import unicode_literals, print_function, division

import spacy

nlp = spacy.load('en')

#extract as list, not set
def ngram_extractor_eval(sent, order):
    ngrams = []
    
    # tokenization
    uwords = [t.text for t in nlp(str(sent))]
    
    # extract ngrams
    for oo in range(1, order + 1):
        for ng in ([' '.join(t).strip() for t in zip(*[uwords[i:] for i in range(oo)])]):
            ngrams.append(ng)
            
    return ngrams


#ROUGE Score, match/reference length
def ROUGE(cand, ref, n):
    cand_ngrams = ngram_extractor_eval(cand, n)
    ref_ngrams = ngram_extractor_eval(ref, n)
    count = 0
    for gram in ref_ngrams:
        if gram in cand_ngrams:
            count += 1
    return count/len(cand_ngrams)

#BLEU Score, match/candidate length, without clipping
def BLEU(cand, ref, n):
    cand_ngrams = ngram_extractor_eval(cand, n)
    ref_ngrams = ngram_extractor_eval(ref, n)
    count = 0
    for gram in cand_ngrams:
        if gram in ref_ngrams:
            count += 1
    return count/len(ref_ngrams)

def BLEU_clip(cand, ref, n):
    cand_ngrams = ngram_extractor_eval(cand, n)
    ref_ngrams = ngram_extractor_eval(ref, n)
    l = len(ref_ngrams)
    count = 0
    for gram in cand_ngrams:
        if gram in ref_ngrams:
            count += 1
            ref_ngrams.remove(gram)
    return count/l

#Coherence score is used for paragraphs, not used yet
def Coherence():
    pass

def score(list_cand, list_ref, n, method='ROUGE'):
    score = 0
    dic = {'ROUGE':ROUGE, 'BLEU':BLEU, 'BLEU_clip':BLEU_clip}
    fun = dic[method]
    num_sent = len(list_cand)
    for i in range(num_sent):
        score += fun(list_cand[i], list_ref[i], n)
    return score/num_sent
