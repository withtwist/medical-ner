from bi_ner_model import Model, Config
from nerloader import NerLoader
import tensorflow as tf
import os, sys
import cPickle
import time
import datetime
import numpy as np
import argparse

def train(instance_name, from_checkpoint = True):
    start_time = time.time()
    conf = Config(instance_name)
    model = Model(conf, false)

    print('save dir:' + conf.save_dir)

    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        train_writer = tf.train.SummaryWriter("save/"+instance_name+"/tensorboard/train", graph_def = sess.graph_def)
        validation_writer = tf.train.SummaryWriter("save/"+instance_name+"/tensorboard/validation")
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=1)
        pointer = 0
        epoch = 0
        if from_checkpoint:
            ckpt = tf.train.get_checkpoint_state(conf.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("{}: Restoring from checkpoint: {}".format(instance_name, ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
                if os.path.exists(os.path.join(conf.save_dir, 'batchpointer.pkl')):
                    with open(os.path.join(conf.save_dir, 'batchpointer.pkl'), 'r') as f:
                        pointer, epoch = cPickle.load(f)
                    print("{}: Continuing from batch number: {}, and epoch: {}".format(instance_name, pointer,epoch))
        conf.nerloader.set_batch_pointer(pointer)
        conf.nerloader.set_epoch(epoch)
        with open(os.path.join(conf.data_dir, 'chars_vocab.pkl'), 'w') as f:
                cPickle.dump((conf.nerloader.chars, conf.nerloader.vocab), f)
            
        for e in xrange(epoch,conf.num_epochs):
            conf.nerloader.set_epoch(e)
            print("Epoch: {}".format(e))
            with open(os.path.join(conf.save_dir, 'batchpointer.pkl'), "w") as f:
                cPickle.dump((conf.nerloader.pointer,conf.nerloader.epoch),f)
            print("{}: Batchpointer: {}, epoch: {}".format(instance_name, conf.nerloader.pointer,conf.nerloader.epoch)) 
            sess.run(tf.assign(model.lr, conf.learning_rate * (conf.decay_rate ** e)))
            for b in xrange(pointer, conf.nerloader.num_batches):       
                start = time.time()
                x, y = conf.nerloader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                train_loss, _ = sess.run([model.cost, model.train_op], feed)
                summary_str = sess.run(model.merged_summary_op, feed_dict=feed)
                train_writer.add_summary(summary_str, e*model.conf.nerloader.num_batches + b)
                end = time.time()
                if b%100 == 0:
                    print instance_name + ": {}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * conf.nerloader.num_batches + b, \
                        conf.num_epochs * conf.nerloader.num_batches,e, train_loss, end - start)
                #Validation
                if b%conf.validate_every == 0 and b != 0:
                    
                    print "{}: {} Computing validation cost".format(instance_name, datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')) 
                    validation_loss_total = 0.0
                    validation_time = time.time()
                    for v in xrange(1 ,conf.nerloader.num_validation_batches):
                        x,y = conf.nerloader.next_validation_batch()
                        validation_feed = {model.input_data: x, model.targets: y}
                        valid_loss = sess.run(model.cost, feed_dict=validation_feed)
                        validation_loss_total += valid_loss
                    avg_valid_loss = validation_loss_total / model.conf.nerloader.num_validation_batches
                    v_summary = tf.scalar_summary('loss', avg_valid_loss)
                    v_m_summary = sess.run(tf.merge_summary([v_summary]))
                    validation_writer.add_summary(v_m_summary,e*model.conf.nerloader.num_batches + b)
                    print "{}: {}: (epoch {}/{}), validation_loss = {:.3f}, validation_test_time = {:.1f})" \
                            .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), instance_name, e, model.conf.num_epochs, avg_valid_loss, time.time()-validation_time)
                    conf.nerloader.reset_validation_pointer()
                  
                if (e * conf.nerloader.num_batches + b) % conf.save_every == 0:
                    checkpoint_path = os.path.join(conf.save_dir, 'model.ckpt')
                
                    saver.save(sess, checkpoint_path, global_step = e * conf.nerloader.num_batches + b)
                    print "{}: model saved to {}".format(instance_name, checkpoint_path)
                    with open(os.path.join(conf.save_dir, 'batchpointer.pkl'), 'w') as f:
                        cPickle.dump((conf.nerloader.pointer,conf.nerloader.epoch), f)
                    print "{}: batchpointer: {} saved.".format(instance_name, conf.nerloader.pointer)
            
                if time.time() - start_time > conf.exit_after*60:
                    checkpoint_path = os.path.join(conf.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * conf.nerloader.num_batches + b) 
                    print "{}: model saved to {}".format(instance_name, checkpoint_path)
                    with open(os.path.join(conf.save_dir, 'batchpointer.pkl'), 'w') as f:
                        cPickle.dump((conf.nerloader.pointer,e), f)
                    print "{}: batchpointer: {} saved.".format(instance_name, conf.nerloader.pointer)
                    print("Has been running for %d seconds. Will exit (exiting after %d minutes)."%((int)(time.time() - start_time), conf.exit_after))
                    return 
            conf.nerloader.reset_batch_pointer()
            pointer = conf.nerloader.pointer
        
        checkpoint_path = os.path.join(conf.save_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step = e * conf.nerloader.num_batches + b) 
        print "model saved to {}".format(checkpoint_path)
        with open(os.path.join(conf.save_dir, 'batchpointer.pkl'), 'w') as f:
            cPickle.dump((conf.nerloader.pointer,e), f)
        print "batchpointer: {} saved.".format(conf.nerloader.pointer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_name', type=str, help='Name of instance')
    parser.add_argument('--from_checkpoint',action='store_true',default=False, help='Continue from checkpoint')
    args = parser.parse_args()
    iname = args.instance_name
    from_checkpoint = args.from_checkpoint
    print "Using instance name: " + iname 
    train(iname,from_checkpoint)

if __name__ == '__main__':
    main()
