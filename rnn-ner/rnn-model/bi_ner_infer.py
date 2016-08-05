from bi_ner_model import Model, Config
from nerloader import NerLoader
import tensorflow as tf
import numpy as np
import time
import codecs
import sys, os
import string
import argparse

def _main(instance_name, infer, infer_text_path):
    o = []
    _probs = []
    conf = Config(instance_name, infer, infer_text_path)
    model = Model(conf)
    def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())     
        ckpt = tf.train.get_checkpoint_state(conf.save_dir)
        print(conf.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continuing from checkpoint...")
            saver.restore(sess, ckpt.model_checkpoint_path)

        #state = model.initial_state.eval()
        state_fw = model.cell_fw.zero_state(1, tf.float32).eval()
        state_bw = model.cell_bw.zero_state(1, tf.float32).eval()

        for b in xrange(conf.nerloader.num_infer_batches):
            start = time.time()
            x = conf.nerloader.next_infer_batch()
            feed = {model.input_data: x, model.initial_state_fw: state_fw, model.initial_state_bw: state_bw}
            [probs] = sess.run([model.probs], feed)
            end = time.time()
            o += [list(p).index(max(p)) for p in probs]
    toSGML(conf, o, conf.nerloader.infer_file, instance_name)
    print "Output written to: save/"+instance_name+"/sgmlout.txt"        


def toSGML(conf, output, test_file, instance_name):
    if os.path.isfile("save/"+instance_name+"/sgmlout.txt"):
        os.remove("save/"+instance_name+"/sgmlout.txt")
    with codecs.open("save/"+instance_name+"/sgmlout.txt", "w", encoding="utf-8") as sgml_file:
        with codecs.open(test_file, encoding="utf-8") as f:
            text = f.read()
        a = zip(list(text),output)
        current_word = []
        previous_wordclass = 0
        for _a in a:
            if isStopChar(_a[0]) and len(current_word) > 0 and (not (_a[1] == classify(current_word)) or _a[1] == 0):
                classes = [w[1] for w in current_word]
                word_string = "".join([w[0] for w in current_word])
                previous_wordclass = wordclass = max(set(classes), key=classes.count)

                print word_string+_a[0]

                print wordclass

                if wordclass == 0:
                    sgml_file.write(word_string+_a[0])

                else:
                    for i, cat in enumerate(conf.nerloader.category_names):
                        if wordclass == i + 1:
                            sgml_file.write("<"+cat+">"+word_string+"</"+cat+">")
                            break
                current_word = []
            elif not isStopChar(_a[0]) or (len(current_word) > 0 and _a[1] == classify(current_word)):
                current_word.append(_a)
            else:
                sgml_file.write(_a[0])
    sgml_file.close()
    return

def classify(word):
    if not word:
        return 0
    classes = [w[1] for w in word]
    return max(set(classes), key=classes.count)

def isStopChar(c):
    return c in [u".",u",",u"(",u"[",u"?",u"!",u"/",u"\n",u"{",u"<",u">",u";",u":", u" "]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_name', type=str, help='Name of instance')
    parser.add_argument('infer_text_path', type=str, help='Path to textfile to classify')
    args = parser.parse_args()
    iname = args.instance_name
    infer_path = args.infer_text_path
    infer = True
    print "Using instance name: {}".format(iname)
    _main(iname, infer, infer_path)
