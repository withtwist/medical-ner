import os.path
import cPickle
import re
from collections import Counter
import codecs

# Generate vocabulary from corpus. 
# Saved in generated files directory
# under the name vocabulary.pkl
# Input:    corpus_path is the path to the corpus file
#           frequency_threshold is the frequency threshold
#           generated_files_path is the path to the directory with generated files where vocabulary will be saved


def generate(corpus_path, frequency_threshold, vocabulary_path):
    corpus_text = ""
    with codecs.open(corpus_path, "r", 'utf-8') as f:
        corpus_text = f.read()

    wordlist = [x.lower() for x in corpus_text.split()]
    
    c = Counter(wordlist)
    vocabulary = []
    for word in set(wordlist):
        if c[word] > frequency_threshold and not(re.match("^[0-9,.,\,,\-,\+,\*,\(,\),%,!,\?,=,<,>,&]*$", word)):
            vocabulary.append(word)
    vocabulary = sorted(vocabulary)

    with open(vocabulary_path, "wb") as f:
        cPickle.dump(vocabulary, f)
