import re


def generate(text_string):
    sentences = re.split("[\.\!\?] ", text_string)
    number_of_sentences = len(sentences)
    sentence_frequency = _count_sentence_frequency(sentences)
    wordlist = _create_wordlist(text_string)
    dictlist = _create_dictlist(wordlist)
    return number_of_sentences, sentence_frequency, wordlist, dictlist


def _create_wordlist(text_string):
    new_text_string = re.sub(r'([\(\)\[\]\{\}\?\!\"\'<>])', r' \1 ', text_string)
    new_text_string = re.sub(r'(\.) ', r' \1', new_text_string)
    new_text_string = re.sub(r'(\,) ', r' \1 ', new_text_string)
    new_text_string = re.sub(r'(\:) ', r' \1 ', new_text_string)
    new_text_string = re.sub(r'(\;) ', r' \1 ', new_text_string)
    new_text_string = re.sub(r'  *', r' ', new_text_string)
    return new_text_string.split()


def _create_dictlist(wordlist):
    dictlist = dict()
    for i, word in enumerate(wordlist):
        if word in dictlist:
            dictlist[word] = dictlist[word] + [i]
        else:
            dictlist[word] = [i]
    return dictlist


def _count_sentence_frequency(sentences):
    sentence_frequency = dict()
    for sent in sentences:
        words = set(_create_wordlist(sent))
        for word in words:
            if word in sentence_frequency:
                sentence_frequency[word] += 1
            else:
                sentence_frequency[word] = 1
    return sentence_frequency
