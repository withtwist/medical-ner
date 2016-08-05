
import tensorflow as tf
from tensorflow.models.rnn import rnn, seq2seq
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell, MultiRNNCell, DropoutWrapper
import numpy as np
import os
import codecs
import time
import cPickle
from IPython.display import display, HTML 
from nerloader import NerLoader


class Model():
    def __init__(self, conf):
        self.conf = conf

        cell_fw = BasicLSTMCell(self.conf.rnn_size)
        cell_bw = BasicLSTMCell(self.conf.rnn_size)
        
        if conf.keep_prob < 1.0 and not conf.infer:
            cell_fw = DropoutWrapper(cell_fw, output_keep_prob=conf.keep_prob)
            cell_bw = DropoutWrapper(cell_bw, output_keep_prob=conf.keep_prob)
        self.cell_fw = cell_fw = MultiRNNCell([cell_fw] * self.conf.num_layers)
        self.cell_bw = cell_bw = MultiRNNCell([cell_bw] * self.conf.num_layers)
        
        self.input_data = tf.placeholder(tf.int32, [self.conf.batch_size, self.conf.seq_length])
        self.targets = tf.placeholder(tf.int32, [self.conf.batch_size, self.conf.seq_length])
    
        self.initial_state_fw = cell_fw.zero_state(self.conf.batch_size, tf.float32)
        
        self.initial_state_bw = cell_bw.zero_state(self.conf.batch_size, tf.float32)
        with tf.variable_scope('rnn'):
            softmax_w = tf.get_variable("softmax_w", [self.conf.rnn_size*2, self.conf.output_size])
            softmax_b = tf.get_variable("softmax_b", [self.conf.output_size])
        
        embedding = tf.get_variable("embedding", [self.conf.nerloader.vocab_size, self.conf.rnn_size])
        _inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if conf.keep_prob < 1.0 and not conf.infer:
            _inputs = tf.nn.dropout(_inputs,conf.keep_prob)
        inputs = tf.split(1, conf.seq_length, _inputs)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            
        outputs_bi = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=self.initial_state_fw, initial_state_bw=self.initial_state_bw, scope='rnn')
        output = tf.reshape(tf.concat(1, outputs_bi), [-1, self.conf.rnn_size*2])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)

        self.loss_weights = [tf.ones([self.conf.batch_size * self.conf.seq_length])]

        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                self.loss_weights)
        self.cost = (tf.reduce_sum(loss) / self.conf.batch_size / self.conf.seq_length)
        tf.scalar_summary("loss",self.cost)
        self.out = output
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                self.conf.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.merged_summary_op = tf.merge_all_summaries()

class Config(object):
    def __init__(self, instance_name, infer, infer_path=""):
    
    
        if os.path.exists("save/" + instance_name +"/conf.pkl"):
            print "Loading config..."
            with open("save/"+instance_name+"/conf.pkl") as f:
                tmp = cPickle.load(f)
            _dict = vars(tmp)
            for x in _dict.keys():
                setattr(self,x, _dict[x])
            self.infer = infer
            if infer:
                self.batch_size = 1
            self.validate_every = 100
            self.nerloader = NerLoader(self.save_dir,self.data_dir,self.batch_size,self.seq_length,self.instance_name, infer, infer_path)
            print self.batch_size
        else:
            self.rnn_size = 128
            self.num_layers = 3

            self.infer = infer
            self.instance_name = instance_name
            
            self.keep_prob = 0.5

            if infer:
                self.batch_size = 1
            else:
                self.batch_size = 30

            print self.batch_size

            self.seq_length = 60

            self.num_epochs = 50

            self.learning_rate = 0.002

            self.decay_rate = 0.975

            self.save_every = 100
            self.validate_every = 100
            self.output_size = 4
            self.grad_clip = 5.0

            self.data_dir = "data"
            self.save_dir = "save/" + instance_name
            if not os.path.exists(self.save_dir):
                print("Creating directory for saving...")
                os.mkdir(self.save_dir)
                
            self.exit_after = 50
            with open(os.path.join(self.save_dir, "conf.pkl"),"w") as f:
                cPickle.dump(self, f) 
            self.nerloader = NerLoader(self.save_dir,self.data_dir,self.batch_size,self.seq_length,self.instance_name, infer, infer_path)



