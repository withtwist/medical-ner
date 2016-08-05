import argparse
import cPickle
import random
import codecs
import numpy as np


g_variants = 0

def main():
    global g_variants
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to file with text input')
    parser.add_argument('target_path', type=str, help="Path to file with targets corresponding to input")
    parser.add_argument('--seq_length', type=int, default=60, help='length of the sequence')
    parser.add_argument('--variants', type=int, default=3, help='number of variants for each occurence of words')
    parser.add_argument('--num_categories', type=int, default=3, help='Number of classification categories')
    parser.add_argument('--no_validation', type=bool, default=False, help='Stop generation of validation set')

    args = parser.parse_args()

    input_path = args.input_path
    target_path = args.target_path

    g_variants = args.variants


    inputs, targets = get_seq(args, input_path, target_path)
    nbr_examples = len(inputs)

    p = np.random.permutation(nbr_examples)

    _inputs = []
    _targets = []
    for k in p:
        _inputs.append(inputs[k])
        _targets.append(targets[k])
    inputs = _inputs
    targets = _targets

    validation_index = int(0.9 * nbr_examples)

    validation_inputs = inputs[validation_index:]
    validation_targets = targets[validation_index:]
    inputs = inputs[:validation_index]
    targets = targets[:validation_index]

    #print "Training:\nInputs: {}\tTargets: {}".format(len(inputs), len(targets))
    #print "Validation:\n Inputs: {}\tTargets: {}.".format(len(validation_inputs), len(validation_targets))


    in_file = codecs.open("data/tf_inputs.txt", "w", encoding="utf-8")
    target_file = codecs.open("data/tf_targets.txt", "w", encoding="utf-8")
    for i in xrange(0,len(inputs)):
        in_file.write(inputs[i])
        target_file.write(''.join(targets[i]))
    in_file.close()
    target_file.close()
    validation_in_file = codecs.open("data/tf_validation_inputs.txt", "w", encoding="utf-8")
    validation_target_file = codecs.open("data/tf_validation_targets.txt", "w", encoding="utf-8")
    for i in xrange(0,len(validation_inputs)):
        validation_in_file.write(validation_inputs[i])
        validation_target_file.write(validation_targets[i])
    validation_in_file.close()
    validation_target_file.close()
    print "Done"
    print "Tensorflow inputs and targets written to data/tf_inputs.txt and data/tf_targets.txt"
    print "Tensorflow  validation inputs and targets written to data/tf_validation_inputs.txt and data/tf_validation_targets.txt"

def get_seq(args, input_path, target_path):
    global g_variants
    input_file = codecs.open(input_path, "r", "utf-8")
    inputs = input_file.read().replace("\n", " ")
    input_file.close()

    y_inputs = []
    y_targets = []

    target_file = open(target_path, "r")
    targets = target_file.read()
    target_file.close()
    
    target_length = len(targets)
    print "Target length: {}".format(target_length)
    seq_length = args.seq_length
    i = 0
    while i < target_length:
        if targets[i] in [str(x) for x in xrange(1, args.num_categories + 1)]:
            w_start = i
            w_end = i
            w_cat = targets[i]
            while targets[w_end+1] == w_cat:
                w_end += 1
            w_size = w_end - w_start + 1
            if w_size < seq_length:
                w_interval = seq_length - w_size

                t_term = "".join(inputs[w_start:w_end+1]).lower()
                
                for variant in xrange(0,g_variants):
                    r = random.randint(0,w_interval)
                    y_start = w_start-r
                    y_end = w_start-r+seq_length
                    if y_start >= 0 and y_end < target_length:
                        y_inputs.append(inputs[y_start:y_end])
                        y_targets.append(targets[y_start:y_end])
            i = w_end
        i += 1


    print "Y-Size: {}".format(len(y_inputs))
    return y_inputs, y_targets

if __name__ == '__main__':
    main()


