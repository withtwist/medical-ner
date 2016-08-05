import cPickle
import subprocess
import xml.etree.ElementTree as ET
import codecs

GATE_PATH = "utils/gate/gate-embedded-processing-1.0.jar"
PIPELINE = "utils/gate/pipeline"


def start(input_path, gate_xml_path, wordlist_path, dictlist_path, pos_tag_doc_path, nbr_sents_path, sent_freq_path):
    parole_sents, wordlist = _processText(input_path, gate_xml_path)

    _generate_dictlist(wordlist, dictlist_path)
    
    number_of_sentences = len(parole_sents)
    sentence_frequency = dict()
    
    with open(wordlist_path, "wb") as f:
        cPickle.dump(wordlist, f)
    wordlist = None

    with open(pos_tag_doc_path, "a") as f:
        for sent in parole_sents:
            notag_sents = ["/".join(notag_w.split("/")[:-1]) for notag_w in sent.lower().split()]
            for nw in set(notag_sents):
                if nw in sentence_frequency:
                    sentence_frequency[nw] = sentence_frequency[nw] + 1
                else:
                    sentence_frequency[nw] = 1
            if isinstance(sent, unicode):
                f.write(sent.encode("utf-8") + "\n")
            else:
                f.write(sent + "\n")

    parole_sents = None

    with open(nbr_sents_path, "wb") as f:
        cPickle.dump(number_of_sentences, f)

    with open(sent_freq_path, "wb") as f:
        cPickle.dump(sentence_frequency, f)


def _processText(input_path, gate_xml_path):
    global GATE_PATH, PIPELINE
    subprocess.call(['java', '-jar', GATE_PATH, PIPELINE, input_path, gate_xml_path])
    return _parseText(gate_xml_path)


def _parseText(gate_xml_path):
    translator = _getTranslator()
    tree = ET.parse(gate_xml_path)
    root = tree.getroot()
    annotations = root[2]

    sents = []
    sent = ""
    endOfSents = []
    splits = []
    wordlist = []
    # Find position of sentences and splits
    listOfAnnotations = []
    for a in annotations:
        a_type = a.attrib["Type"]
        if a_type in ["Token", "SpaceToken"]:
            for f in a:
                if f[0].text == "string":
                    listOfAnnotations.append((int(a.attrib["Id"]),f[1].text))

        if a_type == "Sentence":
            endOfSents.append(a.attrib["EndNode"])
        elif a_type == "Split":
            splits.append((a.attrib["StartNode"], a.attrib["EndNode"]))
    listOfAnnotations.sort(key=lambda tup: tup[0])
    for a in listOfAnnotations:
        wordlist.append(a[1])
    # Find SpaceTokens and Tokens
    # put them in list of sentences with parole tags
    eosCounter = 0
    splCounter = 0
    for a in annotations:
        a_type = a.attrib["Type"]
        if a_type == "SpaceToken" and sent != "":
            sent += " "
        elif a_type == "Token":
            if splCounter < len(splits):
                s = splits[splCounter]
                if s[0] == a.attrib["StartNode"] and s[1] == a.attrib["EndNode"]:
                    sent += " "
                    splCounter += 1
            
            features = dict()
            for f in a:
                features[f[0].text] = f[1].text

            if "string" in features and "category" in features:
                sent += features["string"] + "/" + translator[features["category"].replace("+","/").lower()] + " "
        
        if eosCounter < len(endOfSents) and endOfSents[eosCounter] == a.attrib["EndNode"]:
            eosCounter += 1
            sents.append(sent)
            sent = ""
    return (sents, wordlist)


def _generate_dictlist(wordlist, dictlist_path):
    dictlist = dict()
    for i, word in enumerate(wordlist):
        if word in dictlist:
            dictlist[word] = dictlist[word] + [i]
        else:
            dictlist[word] = [i]
    with open(dictlist_path, "wb") as f:
        cPickle.dump(dictlist, f)


def _getTranslator():

    translator = dict()

    infile = open("utils/suc_tags.csv")

    for line in infile:
        tmp = line[1:-4]
        par, suc = tmp.split(";")
        suc = suc.replace(" ", ".").lower()
        translator[suc] = par
    
    infile.close()

    return translator
