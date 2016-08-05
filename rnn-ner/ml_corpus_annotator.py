
# -*- coding: utf-8 -*-
import cPickle
from multiprocessing import Pool
import multiprocessing
import operator
import numpy as np
import argparse
import os


def _get_wordlist(input_txt):
    sep_wl = input_txt.lower().split(" ")
    wl = [sep_wl[0]]
    for w in sep_wl[1:]:
        wl.append(" ")
        if len(w) > 0:
            if w[0] in [u"(", u"[", u"{", u"<", u'"', u"'", u"`", u"´"]:
                wl.append(w[0])
                w = w[1:]
            if len(w) > 0:
                if w[-1] in [u")", u"]", u"}", u">", u'"', u"'", u"`", u"´", u",", u".", u"-", u"=", u"?", u"!"]:
                    wl.append(w[:-1])
                    wl.append(w[-1])
                else:
                    wl.append(w)
    return wl

def _get_dictlist(wordlist):
    dl = dict()
    for i,w in enumerate(wordlist):
        if w in dl:
            pos_list = dl[w]
            pos_list.append(i)
            dl[w] = pos_list
        else:
            dl[w] = [i]
    return dl


def _get_annotated_category_list(category, wordlist, dictlist):
    category_id, _,  seed_term_list = category
    annotations = []
    for ist, st in enumerate(seed_term_list):

        words = st.lower().split(" ")
        terms = []
        for w in words:
            terms.append(w)
            terms.append(" ")
        terms = terms[:-1]

        if terms[0] in dictlist:
            matches = dictlist[terms[0]]
            for m in matches:
                if wordlist[m:m+len(terms)] == terms:
                    m_start = len("".join(wordlist[0:m]))
                    m_end = m_start + len("".join(terms))
                    annotations.append((category_id, m_start, m_end))
    return annotations

def get_annotations(seedlists, wordlist, dictlist):
    annotations = [] 
    for c in seedlists:
        annotations.append(_get_annotated_category_list(c, wordlist, dictlist))
    annotations = reduce(operator.add, annotations)
    
    return annotations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to file with text input')
    parser.add_argument('seed_path', type=str, help='Path to folder with seed files')

    args = parser.parse_args()

    input_file = args.input_path

    seed_path = args.seed_path

    seedlists = []
    cat_names = open('data/category_names.txt', 'w')
    for _id,seed_list_path in enumerate(os.listdir(seed_path)):
        cat_names.write(os.path.splitext(seed_list_path)[0]+',')
        print "Using ID: {} for {}".format(_id+1,seed_list_path)
        seed_file = open(os.path.join(seed_path, seed_list_path), "r")
        seeds = seed_file.read().decode("utf-8")
        seeds = seeds.split("\n")
        seedlists.append((_id+1,seed_list_path, seeds))
        seed_file.close()

    cat_names.close()
    input_file = open(input_file, "r")
    input_txt = input_file.read().replace("\n", " ")
    input_txt = input_txt.decode("utf-8")
    input_file.close()

    wordlist = _get_wordlist(input_txt)
    dictlist = _get_dictlist(wordlist)

    targets = get_annotations(seedlists, wordlist, dictlist)

    target_file_list = [0] * len(input_txt)

    for tar in targets:
        cat_id = tar[0]
        for x in xrange(tar[1],tar[2]):
            target_file_list[x] = cat_id

    target_file = open("data/corpus_targets.txt", "w")
    for t in target_file_list:
        target_file.write(str(t))
    target_file.close()
    print "Input text targets written to data/corpus_targets.txt"

if __name__ == '__main__':
    main()




