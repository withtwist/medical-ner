import os
import collections
import cPickle
import numpy as np
import codecs
import re

class NerLoader():
    def __init__(self,save_dir, data_dir, batch_size, seq_length, instance_name, infer, infer_path):
        self.instance_name = instance_name	

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        if infer:
            self.infer_file = infer_path

        input_file = os.path.join(data_dir, "tf_inputs.txt")
        target_file = os.path.join(data_dir, "tf_targets.txt")
        vocab_file = os.path.join(save_dir, "chars_vocab.pkl")
        tensor_file = os.path.join(save_dir, "data.npy")
        validation_input = os.path.join(data_dir, "tf_validation_inputs.txt")
        validation_target = os.path.join(data_dir, "tf_validation_targets.txt")

        self.input_file = input_file
        self.target_file = target_file
        
        self.validation_input = validation_input
        self.validation_target = validation_target

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print "Reading text file"
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print "Loading preprocessed files"
            self.load_preprocessed(vocab_file, tensor_file)
        if infer:
            self.create_infer_batches()
            self.reset_infer_pointer()
        else:
            self.create_batches()
            self.create_validation_batches()
            self.reset_validation_pointer()
            self.reset_batch_pointer()
            self.reset_epoch_pointer()

        cat_name_file = os.path.join(data_dir, 'category_names.txt')
        self.category_names = []
        with open(cat_name_file) as f:
            for cat in f.read().split(',')[:-1]:
                self.category_names.append(cat)

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = list(zip(*count_pairs))
        self.vocab_size = len(self.chars)+1
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.none_char = len(self.chars)
        with open(vocab_file, 'w') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(map(lambda x: self.vocab.get(x,self.none_char), data))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.none_char = len(self.chars)
        self.vocab_size = len(self.chars)+1
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

    def create_batches(self):
        with codecs.open(self.input_file, "r", encoding="utf-8") as f:
            x_data = f.read()
        with codecs.open(self.target_file, "r", encoding="utf-8") as f:
            y_data = f.read()
        
        tensor_x = np.array(map(lambda x: self.vocab.get(x,self.none_char),x_data))
        tensor_y = np.array(map(int,y_data))
        self.num_batches = tensor_x.size / (self.batch_size*self.seq_length)
        tensor_x = tensor_x[:self.num_batches * self.batch_size * self.seq_length]
        tensor_y = tensor_y[:self.num_batches * self.batch_size * self.seq_length]
        xs = np.split(tensor_x.reshape(self.batch_size, -1), self.num_batches, 1)
        ys = np.split(tensor_y.reshape(self.batch_size, -1), self.num_batches, 1)
        self.x_batches = xs
        self.y_batches = ys
    def create_infer_batches(self):
        with codecs.open(self.infer_file, "r",encoding="utf-8") as f:
            data = f.read()
        tensor = np.array(map(lambda x: self.vocab.get(x,self.none_char),data))
        self.num_infer_batches = (tensor.size / (self.batch_size*self.seq_length)) + 1 #Due to padding instead of cutting

        chars_to_add = (self.batch_size*self.seq_length) - (tensor.size % (self.batch_size*self.seq_length))

        tensor = np.concatenate([tensor, np.array([self.vocab.get(" ")] * chars_to_add)]) 
        xs = np.split(tensor.reshape(self.batch_size, -1), self.num_infer_batches, 1)
        self.infer_batches = xs
    def create_validation_batches(self):
        with codecs.open(self.validation_input, "r", encoding="utf-8") as f:
            _in = f.read()
            _in = _in.replace('\n', '')
        with codecs.open(self.validation_target, "r", encoding="utf-8") as f:
            _tar = f.read()
            _tar = _tar.replace('\n', '')
        tensor_x = np.array(map(lambda x: self.vocab.get(x, self.none_char), _in))
        tensor_y = np.array(map(int,_tar))
        self.num_validation_batches = tensor_x.size / (self.batch_size*self.seq_length)
        print self.num_validation_batches
        tensor_x = tensor_x[:self.num_validation_batches * self.batch_size * self.seq_length]
        tensor_y = tensor_y[:self.num_validation_batches * self.batch_size * self.seq_length]
        xs = np.split(tensor_x.reshape(self.batch_size, -1), self.num_validation_batches, 1)
        ys = np.split(tensor_y.reshape(self.batch_size, -1), self.num_validation_batches, 1)
        self.validation_x = xs
        self.validation_y = ys
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def next_infer_batch(self):
        x = self.infer_batches[self.infer_pointer]
        self.infer_pointer += 1
        return x
    def next_validation_batch(self):
        x, y = self.validation_x[self.validation_pointer], self.validation_y[self.validation_pointer]
        self.validation_pointer += 1
        return x, y
    def reset_batch_pointer(self):
        self.pointer = 0
    def reset_epoch_pointer(self):
       self.epoch = 0
    def reset_validation_pointer(self):
       self.validation_pointer = 0
    def reset_infer_pointer(self):
        self.infer_pointer = 0
    def next_epoch(self):
       self.epoch += 1
    def set_batch_pointer(self, ptr):
        self.pointer = ptr
    def set_epoch(self, epoch):
       self.epoch = epoch
