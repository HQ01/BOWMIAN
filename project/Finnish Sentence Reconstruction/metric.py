from __future__ import unicode_literals, print_function, division

from utils import *

#ROUGE Score, match/reference length
def ROUGE(cand, ref, n=2):
    cand_ngrams = extract_ngrams(cand, n)
    ref_ngrams = extract_ngrams(ref, n)
    count = 0
    for gram in ref_ngrams:
        if gram in cand_ngrams:
            count += 1
    return count/len(cand_ngrams)

#BLEU Score, match/candidate length, without clipping
def BLEU(cand, ref, n=4):
    cand_ngrams = extract_ngrams(cand, n)
    ref_ngrams = extract_ngrams(ref, n)
    count = 0
    for gram in cand_ngrams:
        if gram in ref_ngrams:
            count += 1
    return count/len(ref_ngrams)

def BLEU_clip(cand, ref, n=4):
    cand_ngrams = extract_ngrams(cand, n)
    ref_ngrams = extract_ngrams(ref, n)
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

def score(list_cand, list_ref, method='ROUGE', n=4):
    score = 0
    dic = {'ROUGE':ROUGE, 'BLEU':BLEU, 'BLEU_clip':BLEU_clip}
    fun = dic[method]
    num_sent = len(list_cand)
    for i in range(num_sent):
        score += fun(list_cand[i], list_ref[i], n)
    return score/num_sent

def multi_score(list_cand, list_ref):
    score_rouge1 = 0
    score_rouge2 = 0
    score_bleu = 0
    score_bleu_clip = 0
    num_sent = len(list_cand)
    for i in range(num_sent):
        score_rouge1 += ROUGE(list_cand[i], list_ref[i], 1)
        score_rouge2 += ROUGE(list_cand[i], list_ref[i], 2)
        score_bleu += BLEU(list_cand[i], list_ref[i])
        score_bleu_clip += BLEU_clip(list_cand[i], list_ref[i])
    score_rouge1 /= num_sent
    score_rouge2 /= num_sent
    score_bleu /= num_sent
    score_bleu_clip /= num_sent
    return score_rouge1, score_rouge2, score_bleu, score_bleu_clip
