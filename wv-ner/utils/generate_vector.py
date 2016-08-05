import cPickle
import numpy as np
import utils.idf as idf
import operator
import codecs


def generate(
        vocabulary_path, wordlist, dictlist, term_list,
        number_of_sentences, sentence_frequency, context_window,
        context_weight, internal_weight, with_weighted_context):
    vocabulary = []
    with open(vocabulary_path, "r") as f:
        vocabulary = cPickle.load(f)

    vector = np.zeros(len(vocabulary))
    nbr_of_vectors = 0
    if with_weighted_context:
        for term in term_list:
            term_vector = _calc_sig_v_weighted(
                vocabulary, wordlist, dictlist, term,
                number_of_sentences, sentence_frequency, context_window,
                context_weight, internal_weight, with_weighted_context)
            if term_vector is not None:
                nbr_of_vectors += 1
                vector = np.add(vector, term_vector)
    else:
        for term in term_list:
            term_vector = _calc_sig_v(
                vocabulary, wordlist, dictlist, term,
                number_of_sentences, sentence_frequency, context_window,
                context_weight, internal_weight, with_weighted_context)
            if term_vector is not None:
                nbr_of_vectors += 1
                vector = np.add(vector, term_vector)
    if nbr_of_vectors > 0:
        vector = vector/nbr_of_vectors
    return vector


def _calc_sig_v_weighted(
        vocabulary, wordlist, dictlist, term,
        number_of_sentences, sentence_frequency, context_window,
        context_weight, internal_weight, with_weighted_context):
    terms = term.rstrip('\n').split()
    matches = []

    if not(terms[0] in dictlist):
        return None
    wordpos = dictlist[terms[0]]

    for i in wordpos:
        curr_wl = wordlist[i:i+len(terms)]
        if curr_wl == terms:
            matches.append(i)
    numberOfMatches = len(matches)
    if numberOfMatches == 0:
        return None

    contextwords = [wordlist[x-context_window:x]+wordlist[x+1+len(terms):x+context_window+1+len(terms)] for x in matches]
    weights = [1.0/x for x in xrange(1, context_window+1)]
    weights = list(reversed(weights))+weights

    cvec = np.zeros(len(vocabulary))
    for clist in contextwords:
        for c,w in zip(clist, weights):
            if c in vocabulary:
                index = vocabulary.index(c)
                cvec[index] = cvec[index] + context_weight*idf.calculate(c, number_of_sentences, sentence_frequency)*w

    for t in term.split():
        if t in vocabulary:
            index = vocabulary.index(t)
            cvec[index] += internal_weight*numberOfMatches*idf.calculate(t, number_of_sentences, sentence_frequency)
    return cvec


def _calc_sig_v(
        vocabulary, wordlist, dictlist, term,
        number_of_sentences, sentence_frequency, context_window,
        context_weight, internal_weight, with_weighted_context):
    terms = term.rstrip('\n').split()
    matches = []

    if not(terms[0] in dictlist):
        return None

    wordpos = dictlist[terms[0]]

    for i in wordpos:
        curr_wl = wordlist[i:i+len(terms)]
        if curr_wl == terms:
            matches.append(i)

    numberOfMatches = len(matches)

    if numberOfMatches == 0:
        return None

    contextwords = reduce(operator.add, [wordlist[x-context_window:x]+wordlist[x+1+len(terms):x+context_window+1+len(terms)] for x in matches], [])
    contextCounter = Counter(contextwords)
    contexts = [(x,contextCounter[x]) for x in set(contextwords)]

    cvec = np.zeros(len(vocabulary))
    for c,i in contexts:
        if c in vocabulary:
            index = vocabulary.index(c)
            cvec[index] = context_weight*i*idf.calculate(c, number_of_sentences, sentence_frequency)
    for t in term.split():
        if t in vocabulary:
            index = vocabulary.index(t)
            cvec[index] += internal_weight*numberOfMatches*idf.calculate(t, number_of_sentences, sentence_frequency)
    return cvec

