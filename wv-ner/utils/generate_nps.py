import os
import cPickle
import re
import codecs
import idf


def generate(pos_tag_doc_path, generated_files_path, np_tag_doc_path, nps_path, number_of_sentences, sentence_frequency, idf_threshold):
    os.system("cd utils/spark; python run_parse.py -p 1 -1 ../../{} ../../{}".format(pos_tag_doc_path, np_tag_doc_path))
    os.system("cd ../..")

    np_string = ""
    with codecs.open(np_tag_doc_path, "r", 'utf-8') as f:
        np_string = f.read()
    
    np_tags = re.findall("\[NP\*(.*?)\*NP\]", np_string)
    nphrases = [_remove_tags(nphrase).lower() for nphrase in np_tags]    
    filtered_nps = [nphrase for nphrase in nphrases if idf.calculate(nphrase, number_of_sentences, sentence_frequency) > idf_threshold]

    with open(nps_path, "wb") as f:
        cPickle.dump(filtered_nps, f)

    return filtered_nps


def _remove_tags(nphr):
    nphr = re.sub("\[[A-Z]*?\*\ ", "", nphr)
    nphr = re.sub("\ \*[A-Z]*?\]", "", nphr)
    nphr = re.sub("[ ]{2,}", " ", nphr).strip()

    terms = nphr.split(" ")
    terms = ["/".join(x.split("/")[:-1]) for x in terms]

    return " ".join(terms)
