import numpy as np
import multiprocessing
from multiprocessing import Pool
import operator
import idf
import math
import generate_vector as gen_vec
import cPickle

UNCLASSIFIED_ID = -1

# Classifies the entity candidates
# into one of the predefined categories
# Input:    entities -  is a list of entities to classify
#           wordlist_local - is the wordlist
#           vocabulary_local - is the vocabulary
#           args_local - is the parameters
#           category_local - is the category vectors


def classify(
        entities, wordlist_path, dictlist_path, 
        vocabulary_path, categories, nbr_sents_path, 
        sent_freq_path, context_window, context_weight,
        internal_weight, with_weighted_context, nbr_of_threads,
        classification_threshold, similarity_threshold):
    wordlist = []
    with open(wordlist_path, "rb") as f:
        wordlist = cPickle.load(f)

    dictlist = dict()
    with open(dictlist_path, "rb") as f:
        dictlist = cPickle.load(f)

    number_of_sentences = 0
    with open(nbr_sents_path, "r") as f:
        number_of_sentences = cPickle.load(f)

    sentence_frequency = dict()
    with open(sent_freq_path, "r") as f:
        sentence_frequency = cPickle.load(f)

    if nbr_of_threads == 1:
        classified_entities = _classify_entities_batch(
            entities, wordlist, dictlist,
            vocabulary_path, categories, number_of_sentences,
            sentence_frequency, context_window, context_weight,
            internal_weight, with_weighted_context,
            classification_threshold, similarity_threshold)
    else:
        if nbr_of_threads == 0:
            nbr_of_threads = multiprocessing.cpu_count()
        npchunks = np.array_split(np.array(entities), nbr_of_threads)
        pool = Pool(nbr_of_threads)
        try:
            chunks = [[
                chunk, wordlist, dictlist,
                vocabulary_path, categories, number_of_sentences,
                sentence_frequency, context_window, context_weight,
                internal_weight, with_weighted_context,
                classification_threshold, similarity_threshold] for chunk in list(npchunks)]
            classified_entities = reduce(operator.add, pool.map(_classify_entities_batch_star, chunks))
        except:
            pool.terminate()
            pool.close()
            raise ValueError('Error occurred while classifying in thread.')

        pool.terminate()
        pool.close()

    return classified_entities


def _classify_entities_batch_star(p):
    return _classify_entities_batch(
        p[0], p[1], p[2], p[3],
        p[4], p[5], p[6], p[7],
        p[8], p[9], p[10], p[11],
        p[12])


def _classify_entities_batch(
        nps, wordlist, dictlist,
        vocabulary_path, categories, number_of_sentences,
        sentence_frequency, context_window, context_weight,
        internal_weight, with_weighted_context,
        classification_threshold, similarity_threshold):
    classified_entities = []
    latest_word = 0
    wordlist_length = len(wordlist)

    for nphrase in nps:
        sig = gen_vec.generate(
            vocabulary_path, wordlist, dictlist, [nphrase],
            number_of_sentences, sentence_frequency, context_window,
            context_weight, internal_weight, with_weighted_context)
        if not (sig is None) and len(sig) != 0:
            (category_id, sims) = _classify_entity(sig, categories, classification_threshold, similarity_threshold)
            if category_id != UNCLASSIFIED_ID:
                np_terms = nphrase.split(" ")
                np_term_length = len(np_terms)
                for i in xrange(latest_word, wordlist_length):
                    if wordlist[i:i + np_term_length] == np_terms:
                        classified_entities.append((category_id, i, i + np_term_length - 1, sims))
                        latest_word = i + np_term_length
                        break
    return classified_entities


def _classify_entity(entityV, categories, classification_threshold, similarity_threshold):
    if len(entityV) == 0:
        return (UNCLASSIFIED_ID, ["?%", "?%", "?%"])
    else:
        similarities = [abs(similarity(entityV, cV)) for cV in categories]
        if any([math.isnan(s) for s in similarities]) or all([s == 0.0 for s in similarities]):
            return (UNCLASSIFIED_ID, ["0%", "0%", "0%"])
        max_val = max(similarities)
        max_id = similarities.index(max_val)

        simsum = sum(similarities)
        normsim = [str(int((currsim/simsum)*100)) + "%" for currsim in similarities]
        for i,s in enumerate(similarities):
            if i != max_id and (s/float(max_val)) > classification_threshold:
                return (UNCLASSIFIED_ID, normsim)
        if max_val > similarity_threshold:
            return (max_id, normsim)
        else:
            return (UNCLASSIFIED_ID, normsim)


def similarity(entityV, categoryV):
    if entityV is None:
        return 0.0
    numerator = (np.dot(entityV, categoryV))
    denominator = (math.sqrt(entityV.dot(entityV))*math.sqrt(categoryV.dot(categoryV)))
    if denominator == 0 or denominator == 0.0:
        return 0.0
    return numerator/float(denominator)
