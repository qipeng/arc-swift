import tensorflow as tf

import argparse
import cPickle as pickle
from smart_open import smart_open
import logging
import sys
import os.path as op
import numpy as np
from utils import *
import random
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid

from parser import Parser
from init_argparse import init_argparse
from itertools import izip
from heapq import nlargest, heappush, heappushpop, nsmallest

import socket
hostname = socket.gethostname().split('.')[0]

tf.logging.set_verbosity(tf.logging.WARN)
logging.basicConfig(format="%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)

def train(args):
    transsys = transsys_lookup(args.transsys)

    vocab, vecs, pretrained = read_vocab(conll_file=args.conll_file, wordvec_file=args.wordvec_file, vocab_file=args.vocab_file, wordvec_dim=args.wordvec_dim, min_count=args.min_count, log=log)
    mappings, invmappings = read_mappings(args.mappings_file, transsys, log=log)
    data, sent_length, trans_length = read_data(conll_file=args.conll_file, seq_file=args.seq_file, vocab=vocab, mappings=mappings, transsys=transsys, fpos=args.fpos, log=log)
    feats, feat_dim = featurize_transitions(data, mappings, invmappings, args.feat_file, transsys, log=log)

    feat_shape = [5] if args.transsys != 'ASw' else [feat_dim, 5]

    parser = Parser(args, vecs, pretrained, mappings, invmappings, sent_length, trans_length, feat_dim, log)
    loss = parser.loss

    log.info('Computational graph successfully built.')
    log.info('Setting up tensorflow session...')

    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    global_step = tf.Variable(0, trainable=False)
    if args.anneal >= 0:
        boundaries = [int(len(data) / args.batch_size * i * args.epoch_multiplier) for i in xrange(args.anneal, args.epochs)]
        values = [args.lr] + [args.lr * (2 ** (args.anneal - i - 1)) for i in xrange(args.anneal, args.epochs)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        learning_rate = args.lr

    if args.opt == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=args.beta2)
    elif args.opt == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    opt_op = opt.minimize(loss, var_list=trainable_variables, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
    init_op = tf.global_variables_initializer()

    UNKID = vocab['<UNK>']

    log.info('Starting tensorflow session...')
    # Avoid taking up all of the memory on the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    epochs = args.epochs
    savefilename = '%s/model_epoch%d'
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        epoch0 = -1
        loaded = False
        for epoch in reversed(xrange(int(epochs * args.epoch_multiplier))):
            filename = savefilename % (args.model_dir, epoch)
            if not op.exists(filename + '.meta'):
                continue
            saver.restore(sess, filename)
            log.info('Previously trained model recovered from "%s"' % (filename))
            epoch0 = epoch
            loaded = True
            sess.run(global_step.assign(len(data) / args.batch_size * (epoch0 + 1)))
            break

        indices = range(len(data))
        for epoch in xrange(epoch0+1, int(epochs * args.epoch_multiplier)):
            random.shuffle(indices)
            for batch in xrange(len(data) / args.batch_size):
                idx = indices[batch * args.batch_size:(batch+1) * args.batch_size]

                batch_data = [data[i] for i in idx]
                batch_feats = [feats[i] for i in idx]

                # prepare data in tensor shape
                batch_sent_lengths = np.array([len(datum[0]) for datum in batch_data], dtype=np.int32)
                batch_words = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                batch_words2 = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                batch_gold_pos = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                if args.fpos:
                    batch_gold_fpos = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                for i in xrange(args.batch_size):
                    batch_words[i, :batch_sent_lengths[i]] = batch_data[i][0]
                    batch_words2[i, :batch_sent_lengths[i]] = batch_data[i][0]
                    batch_gold_pos[i, :batch_sent_lengths[i]] = batch_data[i][2]
                    if args.fpos:
                        batch_gold_fpos[i, :batch_sent_lengths[i]] = batch_data[i][3]

                if args.word_dropout > 0:
                    mask = np.random.rand(*batch_words.shape) <= args.word_dropout
                    batch_words[mask] = UNKID

                batch_trans_feat_ids = np.zeros(tuple([args.batch_size, trans_length] + feat_shape), dtype=np.int32)
                batch_trans_feat_sizes = np.zeros((args.batch_size, trans_length), dtype=np.int32)
                batch_trans_labels = np.zeros((args.batch_size, trans_length), dtype=np.int32)
                batch_trans_lengths = np.array([len(feat[0]) for feat in batch_feats], dtype=np.int32)

                if args.transsys == 'ASw':
                    for i in xrange(args.batch_size):
                        for j in xrange(batch_trans_lengths[i]):
                            batch_trans_feat_ids[i, j, :len(batch_feats[i][0][j]), :] = batch_feats[i][0][j][:len(batch_feats[i][0][j])]
                            batch_trans_feat_sizes[i, j] = batch_feats[i][1][j]
                            assert(batch_feats[i][1][j] > 0)
                            batch_trans_labels[i, j] = batch_feats[i][2][j]
                else:
                    for i in xrange(args.batch_size):
                        for j in xrange(batch_trans_lengths[i]):
                            batch_trans_feat_ids[i, j, :] = batch_feats[i][0][j]
                            batch_trans_feat_sizes[i, j] = batch_feats[i][1][j]
                            assert(batch_feats[i][1][j] > 0)
                            batch_trans_labels[i, j] = batch_feats[i][2][j]

                # perform update
                feed_dict={parser.words: batch_words,
                           parser.words2: batch_words2,
                           parser.sent_lengths: batch_sent_lengths,
                           parser.trans_feat_ids: batch_trans_feat_ids,
                           parser.trans_feat_sizes: batch_trans_feat_sizes,
                           parser.trans_labels: batch_trans_labels,
                           parser.trans_lengths: batch_trans_lengths,
                           parser.gold_pos: batch_gold_pos}
                if args.fpos:
                    feed_dict[parser.gold_fpos] = batch_gold_fpos
                _, l = sess.run([opt_op, loss], feed_dict=feed_dict)

                log.info('Epoch %3d batch %4d Loss: %8f (per transition: %8f)' % (epoch, batch, l, (l * args.batch_size / np.sum(batch_trans_lengths))))

            saver.save(sess, savefilename % (args.model_dir, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train parsers')

    init_argparse(parser)

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    train(args)
