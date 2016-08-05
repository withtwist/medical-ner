import math


def calculate(term, number_of_sentences, sentence_frequency):
    words = term.split()
    idfsum = 0
    for word in words:
        if word in sentence_frequency:
            idfsum += math.log(float(number_of_sentences)/sentence_frequency[word])
        else:
            idfsum += 0
    return float(idfsum)/len(words)
