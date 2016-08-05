import logging
import argparse
import os
import datetime
import time
import cPickle
import codecs
import multiprocessing
import utils.generate_corpus_data as gen_corpus_data
import utils.generate_vector as gen_vector
import utils.generate_vocabulary as gen_vocabulary
import utils.pos_tagger as pos_tagger
import utils.generate_nps as gen_nps
import utils.classifier as classifier
import utils.add_classifications as add_classifications


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('Start running the model')
    start_time = time.time()

    # Parsing the arguments given to the input
    logging.info('Parse arguments: Starting')
    parser = argparse.ArgumentParser()

    # Optional arguments for the parameters of the model
    parser.add_argument('--with_weighted_context', type=bool, default=True, help='whether or not the context words are weighted by distance')
    parser.add_argument('--internal_weight', type=int, default=20, help='weight of internal words, used in relation to context weights')
    parser.add_argument('--context_weight', type=int, default=1, help='weight of context words, used in relation to internal weights')
    parser.add_argument('--frequency_threshold', type=int, default=10, help='number of times a word must occur in the corpus in order to be put in the vocabulary')
    parser.add_argument('--classification_threshold', type=float, default=0.7, help='determines how much greater the similarity between the category that the entity is most similar to must be in comparison to the other categories')
    parser.add_argument('--similarity_threshold', type=float, default=0.005, help='determines how similar the vectors must be in order to be classified at all')
    parser.add_argument('--idf_threshold', type=int, default=2, help='determines how great the IDF value must be for a given entity candidate in order to still be considered as a candidate')
    parser.add_argument('--context_window', type=int, default=5, help='is the number of words on each side of the internal words that will be be considered as context words')
    parser.add_argument('--clean_directory', type=bool, default=False, help='removes all the files in the output directory if there are any')
    parser.add_argument('--nbr_of_threads', type=int, default=1, help='number of threads to use when classifying entities, where 0 is the maximum number of cores available (be aware of the overhead when running multi-threaded)')

    # Required arguments in order for the model to work
    parser.add_argument('--seed_term_lists_path', type=str, default='-1', help='path to directory where all seed term list are, one seed term list for each textfile, separated by new line')
    parser.add_argument('--corpus_path', type=str, default='-1', help='path to corpus that will be used to generate category vectors and vocabulary')
    parser.add_argument('--output_path', type=str, default='-1', help='path to directory where output and generated files will be')
    parser.add_argument('--input_path', type=str, default='-1', help='path to input file that should be classified')

    args = parser.parse_args()

    # Check that the required args are valid
    _check_required_args(args)

    # Check that the optional args are valid
    _check_optional_args(args)

    # Assign values
    with_weighted_context = args.with_weighted_context
    internal_weight = args.internal_weight
    context_weight = args.context_weight
    frequency_threshold = args.frequency_threshold
    classification_threshold = args.classification_threshold
    similarity_threshold = args.similarity_threshold
    idf_threshold = args.idf_threshold
    context_window = args.context_window
    clean_directory = args.clean_directory
    nbr_of_threads = args.nbr_of_threads

    seed_term_lists_path = args.seed_term_lists_path
    corpus_path = args.corpus_path
    input_file_path = args.input_path
    output_path = args.output_path

    # Create directory for generate files
    generated_files_path = output_path + '/generated_files'
    if clean_directory:
        os.system("rm -r {}/*".format(output_path))
    os.system("mkdir {}".format(generated_files_path))

    logging.info('Parse arguments: Done')

    # Generate text file with settings
    logging.info('Generate Settings File: Starting')
    _generate_settings_file(output_path, args)
    logging.info('Generate Settings File: Done')

    # Generate vocabulary from corpus. 
    logging.info('Generate Vocabulary: Starting')
    vocabulary_path = "{}/vocabulary.pkl".format(generated_files_path)
    gen_vocabulary.generate(corpus_path, frequency_threshold, vocabulary_path)
    logging.info('Generate Vocabulary: Done')

    # Generate category vectors for each category in seed term lists path
    number_of_sentences = 0
    sentence_frequency = dict()
    wordlist = []
    dictlist = dict()
    with codecs.open(corpus_path, "r", 'utf-8') as f:
        number_of_sentences, sentence_frequency, wordlist, dictlist = gen_corpus_data.generate(f.read())

    category_names = _generate_category_vectors(
        seed_term_lists_path, vocabulary_path, wordlist,
        dictlist, number_of_sentences, sentence_frequency,
        context_window, context_weight, internal_weight,
        with_weighted_context, generated_files_path)

    wordlist = None
    dictlist = None
    number_of_sentences = None
    sentence_frequency = None

    # Tag input document with POS tags.
    logging.info('POS Tagger: Starting')
    gate_doc_path = '{}/gate_doc.txt'.format(generated_files_path)
    wordlist_path = '{}/wordlist.pkl'.format(generated_files_path)
    dictlist_path = '{}/dictlist.pkl'.format(generated_files_path)
    pos_tag_doc_path = '{}/pos_tag_doc.txt'.format(generated_files_path)
    nbr_sents_path = '{}/global_nbrSents.pkl'.format(generated_files_path)
    sent_freq_path = '{}/global_sentFreq.pkl'.format(generated_files_path)
    pos_tagger.start(input_file_path, gate_doc_path, wordlist_path, dictlist_path, pos_tag_doc_path, nbr_sents_path, sent_freq_path)
    logging.info('POS Tagger: Done')

    number_of_sentences = 0
    with open(nbr_sents_path, "r") as f:
        number_of_sentences = cPickle.load(f)
    sentence_frequency = dict()
    with open(sent_freq_path, "r") as f:
        sentence_frequency = cPickle.load(f)

    # Find noun phrases amongst the words and phrases in the input document
    logging.info('Find Entity Candidates: Starting')
    nps_path = '{}/nps.pkl'.format(generated_files_path)
    np_tag_doc_path = '{}/np_tag_doc.txt'.format(generated_files_path) 
    
    nps = gen_nps.generate(pos_tag_doc_path, generated_files_path, np_tag_doc_path, nps_path, number_of_sentences, sentence_frequency, idf_threshold)
    logging.info('Find Entity Candidates: Done')
    logging.info('Found {} noun phrases'.format(len(nps)))

    logging.info('Load Category Vectors: Starting')
    category_vectors = []
    for c_name in category_names:
        with open("{}/{}.pkl".format(generated_files_path, c_name), "rb") as f:
            category_vectors.append(cPickle.load(f))
    logging.info('Load Category Vectors: Done')

    # Classify entities in input document, given the noun phrases
    logging.info('Classify Entities: Starting')
    wordlist = []
    with open(wordlist_path, "rb") as f:
        wordlist = cPickle.load(f)

    dictlist = dict()
    with open(dictlist_path, "rb") as f:
        dictlist = cPickle.load(f)

    classified_entities = classifier.classify(
        nps, wordlist_path, dictlist_path, 
        vocabulary_path, category_vectors, 
        nbr_sents_path, sent_freq_path,
        context_window, context_weight,
        internal_weight, with_weighted_context,
        nbr_of_threads, classification_threshold,
        similarity_threshold)
    logging.info('Classify Entities: Done')
    logging.info('Classified {} entities'.format(len(classified_entities)))

    # Save classification positions into a file
    logging.info('Generate Output Position: Starting')
    output_position_path = "{}/output_pos.txt".format(output_path)
    _generate_output_position_file(classified_entities, output_position_path)
    logging.info('Generate Output Position: Done')

    output_pos_list = []
    with open(output_position_path, "r") as f:
        output_pos_list = f.read().split("\n")

    with open(wordlist_path, "rb") as f:
        wordlist = cPickle.load(f)

    # Generate output file by adding classifications to input document
    logging.info('Add Classifications: Starting')
    classified_doc = add_classifications.get(output_pos_list, category_names, wordlist)
    logging.info('Add Classifications: Done')

    logging.info('Save Classified Document: Start')
    with open("{}/classified_document.txt".format(output_path), "w") as f:
        f.write(classified_doc.encode('utf-8'))
    logging.info('Save Classified Document: End')

    end_time = time.time()
    logging.info("Elapsed time: {}s".format(int(end_time-start_time)))
    logging.info("Done Classifying Entities")


# Checks if the required arguments were provided and valid
# Input: argument for argparse
def _check_required_args(args):
    if args.seed_term_lists_path == '-1':
        raise ValueError('no argument was provided for seed_term_list_path')
    if not os.path.isdir(args.seed_term_lists_path):
        raise IOError('directory not found for seed term list')
    elif len(os.listdir(args.seed_term_lists_path)) == 0:
        raise IOError('no seed term lists found in directory')

    if args.corpus_path == '-1':
        raise ValueError('no argument was provided for corpus_path')
    if not os.path.isfile(args.corpus_path):
        raise IOError('corpus file do not exist')

    if args.output_path == '-1':
        raise ValueError('no argument was provided for output_path')
    if not os.path.isdir(args.seed_term_lists_path):
        raise IOError('directory not found for output')

    if args.input_path == '-1':
        raise ValueError('no argument was provided for input_path')
    if not os.path.isfile(args.input_path):
        raise IOError('input file do not exist')


def _check_optional_args(args):
    if args.with_weighted_context == None:
        raise ValueError('argument with_weighted_context must be either True or False')
    if not(args.internal_weight > 0):
        raise ValueError('argument internal_weight must be greater than 0')
    if not(args.context_weight > 0):
        raise ValueError('argument context_weight must be greater than 0')
    if not(args.frequency_threshold >= 0):
        raise ValueError('argument frequency_threshold must be greater than or equal to 0')
    if not(args.classification_threshold <= 1.0 and args.classification_threshold >= 0.0):
        raise ValueError('argument classification_threshold must be in the interval 0.0 and 1.0')
    if not(args.similarity_threshold <= 1.0 and args.similarity_threshold >= 0.0):
        raise ValueError('argument similarity_threshold must be in the interval 0.0 and 1.0')
    if not(args.idf_threshold >= 0):
        raise ValueError('argument idf_threshold must be greater than or equal to 0')
    if not(args.context_window >= 0):
        raise ValueError('argument context_window must be greater than or equal to 0')
    if args.clean_directory == None:
        raise ValueError('argument clean_directory must be either True or False')
    if not(args.nbr_of_threads >= 0):
        raise ValueError('argument nbr_of_threads must be greater than or equal to 0')


def _generate_settings_file(output_path, args):
    settings_file = open("{}/settings.txt".format(output_path), "wa")

    settings_file.write("SETTINGS\n")
    settings_file.write("=======================\n")
    settings_file.write("Date: {}".format(datetime.datetime.now()))
    settings_file.write("\n\n")

    settings_file.write("Context Weight: {}\n".format(args.context_weight))
    settings_file.write("Internal Weight: {}\n".format(args.internal_weight))
    settings_file.write("Frequency Threshold: {}\n".format(args.frequency_threshold))
    settings_file.write("Classification Threshold: {}\n".format(args.classification_threshold))
    settings_file.write("Similarity Threshold: {}\n".format(args.similarity_threshold))
    settings_file.write("IDF Threshold: {}\n".format(args.idf_threshold))
    settings_file.write("Context Window: {}\n".format(args.context_window))
    if args.with_weighted_context:
        settings_file.write("Weigthed Context: ON\n")
    else:
        settings_file.write("Weigthed Context: OFF\n")
    if args.clean_directory:
        settings_file.write("Context Weight: ON\n")
    else:
        settings_file.write("Context Weight: OFF\n")
    if args.nbr_of_threads == 0:
        settings_file.write("Number of threads: {} (cores)".format(multiprocessing.cpu_count()))
    else:
        settings_file.write("Number of threads: {}".format(args.nbr_of_threads))
    settings_file.write("\n\n")

    settings_file.write("Seed term list path: {}\n".format(args.seed_term_lists_path))
    settings_file.write("Corpus path: {}\n".format(args.corpus_path))
    settings_file.write("Output path: {}\n".format(args.output_path))
    settings_file.write("Input path: {}".format(args.input_path))

    settings_file.close()


def _generate_category_vectors(
        seed_term_lists_path, vocabulary_path, wordlist,
        dictlist, number_of_sentences, sentence_frequency,
        context_window, context_weight, internal_weight,
        with_weighted_context, generated_files_path):
    list_file_paths = os.listdir(seed_term_lists_path)
    category_names = []
    for list_file_path in list_file_paths:
        with codecs.open("{}/{}".format(seed_term_lists_path, list_file_path), "r", 'utf-8') as f:
            category_name = list_file_path.rstrip('.txt')
            if category_name in category_names:
                i = 2
                while '{}{}'.format(category_name, i) in category_names:
                    i += 1
                category_name += str(i)

            category_names.append(category_name)
            category_list = f.read().split()

        logging.info('Generate category vector {}: Starting'.format(category_name))
        category_vector = gen_vector.generate(
            vocabulary_path, wordlist, dictlist, category_list,
            number_of_sentences, sentence_frequency, context_window,
            context_weight, internal_weight, with_weighted_context)
        
        with open("{}/{}.pkl".format(generated_files_path, category_name), "wb") as f:
            cPickle.dump(category_vector, f)
        logging.info('Generate category vector {}: Done'.format(category_name))
    return category_names

def _generate_output_position_file(classified_entities, output_position_path):
    with open(output_position_path, "a") as f:
        f.write("Class,StartWord,EndWord\n")
        for e in classified_entities:
            f.write("{},{},{}\n".format(e[0], e[1], e[2]))


if __name__ == '__main__':
    main()
