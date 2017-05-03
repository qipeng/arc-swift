import tensorflow as tf

import argparse
import cPickle as pickle
from smart_open import smart_open
import logging
import sys
import os.path as op
import numpy as np

from utils import process_example, read_data, read_vocab, read_mappings, featurize_state
from layers import DenseLayer
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid
from parserstate import ParserState

from parser import Parser
from init_argparse import init_argparse

from time import time
from heapq import nlargest, heappush, heappushpop, nsmallest
from itertools import izip

floatX = np.float32

tf.logging.set_verbosity(tf.logging.WARN)
logging.basicConfig(format="%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)

def eval(args):
    transsys_lookup = {"ASw": ArcSwift,
                       "AER" : ArcEagerReduce,
                       "AES": ArcEagerShift,
                       "ASd"  : ArcStandard,
                       "AH"  : ArcHybrid,}
    transsys = transsys_lookup[args.transsys]

    vocab, vecs, pretrained = read_vocab(conll_file=args.conll_file, wordvec_file=args.wordvec_file, vocab_file=args.vocab_file, wordvec_dim=args.wordvec_dim, min_count=args.min_count, log=log)
    mappings, invmappings = read_mappings(args.mappings_file, transsys, log=log)
    data, sent_length, trans_length = read_data(conll_file=args.conll_file, seq_file=args.seq_file, vocab=vocab, mappings=mappings, transsys=transsys, fpos=args.fpos, log=log)

    feat_shape = [5] if args.transsys != 'ASw' else [sent_length, 5]

    transsys = transsys(mappings, invmappings)

    parser = Parser(args, vecs, pretrained, mappings, invmappings, sent_length, trans_length, -1, log, train=False)

    trans_predictors = parser.trans_predictors

    log.info('Computational graph successfully built.')
    log.info('Setting up tensorflow session...')

    saver = tf.train.Saver(max_to_keep=10000)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    for epoch in reversed(xrange(int(args.epochs * args.epoch_multiplier))):
        #with tf.Session(config=config) as sess:
        savedpath = '%s/model_epoch%d' % (args.model_dir, epoch)
        if not op.exists(savedpath + '.meta'):
            continue
        log.info('Evaluating Epoch %3d...' % (epoch))
        saver.restore(sess, savedpath)

        states = [[(0, ParserState(datum[0], transsys=transsys))] for datum in data]
        with smart_open('%s/%s_pos_eval_beam_%d_output_epoch%d.txt' % (args.model_dir, args.eval_dataset, args.beam_size, epoch), 'w') as outf2:
            with smart_open('%s/%s_eval_beam_%d_output_epoch%d.txt' % (args.model_dir, args.eval_dataset, args.beam_size, epoch), 'w') as outf:
                for batch in xrange((len(data)+args.batch_size-1) / args.batch_size):
                    idx = range(batch * args.batch_size, min((batch+1) * args.batch_size, len(data)))

                    batch_size = len(idx)

                    batch_data = [data[i] for i in idx]
                    batch_states = [states[i] for i in idx]

                    # prepare data in tensor shape
                    batch_sent_lengths = np.array([len(datum[0]) for datum in batch_data] + [sent_length] * (args.batch_size - batch_size), dtype=np.int32)
                    batch_words = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                    batch_words2 = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                    batch_gold_pos = np.zeros((args.batch_size, sent_length), dtype=np.int32)
                    for i in xrange(batch_size):
                        batch_words[i, :batch_sent_lengths[i]] = batch_data[i][0]
                        batch_words2[i, :batch_sent_lengths[i]] = batch_data[i][0]
                        batch_gold_pos[i, :batch_sent_lengths[i]] = batch_data[i][2]

                    batch_trans_feat_ids = np.zeros(tuple([args.batch_size * args.beam_size] + feat_shape), dtype=np.int32)
                    batch_trans_feat_sizes = np.zeros((args.batch_size * args.beam_size), dtype=np.int32)

                    preds_list = [parser.combined_head, parser.combined_dep, parser.pos_preds]
                    if args.transsys == 'ASw':
                        preds_list += [parser.transition_logit]
                    if args.fpos:
                        preds_list += [parser.fpos_preds]

                    preds = sess.run(preds_list,
                               feed_dict={parser.words: batch_words,
                                          parser.words2: batch_words2,
                                          parser.sent_lengths: batch_sent_lengths,
                                          parser.gold_pos: batch_gold_pos,})
                    # unpack predictions
                    batch_combined_head, batch_combined_dep, pos_preds = preds[:3]
                    preds = preds[3:]
                    if args.transsys == 'ASw':
                        batch_trans_logit = preds[0]
                        preds = preds[1:]
                    if args.fpos:
                        fpos_preds = preds[0]
                        preds = preds[1:]

                    if args.fpos:
                        for i in xrange(batch_size):
                            for j in xrange(batch_sent_lengths[i]-1):
                                outf2.write("%s\t%s\n" % (invmappings['pos'][pos_preds[i][j]], invmappings['fpos'][fpos_preds[i][j]]))
                            outf2.write("\n")
                    else:
                        for i in xrange(batch_size):
                            for j in xrange(batch_sent_lengths[i]-1):
                                outf2.write("%s\t_\n" % invmappings['pos'][pos_preds[i][j]])
                            outf2.write("\n")

                    j = 0
                    updated = range(batch_size)
                    batch_finished = [[] for _ in range(batch_size)]

                    feat_lengths = [[] for _ in range(batch_size)]

                    while True:
                        batch_feats = [[featurize_state(batch_states[i][k][1], mappings) for k in range(len(batch_states[i]))] for i in updated]
                        for i, beam_feats in zip(updated, batch_feats):
                            feats = beam_feats[0]
                            if len(feats) > 0:
                                if args.transsys == 'ASw':
                                    feat_lengths[i] += [len(feats)]
                                else:
                                    feat_lengths[i] += [len(batch_states[i][0][1].transitionset())]

                        preds = []
                        predsid = []
                        for i, beam_feats in zip(updated, batch_feats):
                            for k, feats in enumerate(beam_feats):
                                if len(feats) <= 0:
                                    if len(batch_finished[i]) < args.beam_size:
                                        heappush(batch_finished[i], batch_states[i][k])
                                    else:
                                        heappushpop(batch_finished[i], batch_states[i][k])

                                    continue

                                beamidx = i * args.beam_size + k
                                if args.transsys == 'ASw':
                                    batch_trans_feat_ids[beamidx, :len(feats)] = feats
                                else:
                                    batch_trans_feat_ids[beamidx] = feats

                                batch_trans_feat_sizes[beamidx] = len(feats)

                                assert(batch_trans_feat_sizes[beamidx] > 0)

                                predsid.append((i, k))
                                preds.append(trans_predictors[i][k])

                        if len(predsid) <= 0:
                            break

                        if args.transsys == 'ASw':
                            p = sess.run(preds, feed_dict={parser.combined_head_placeholder: batch_combined_head,
                                                       parser.combined_dep_placeholder: batch_combined_dep,
                                                       parser.trans_logit_placeholder:batch_trans_logit,
                                                       parser.trans_feat_ids: batch_trans_feat_ids,
                                                       parser.trans_feat_sizes: batch_trans_feat_sizes})
                        else:
                            p = sess.run(preds, feed_dict={parser.combined_head_placeholder: batch_combined_head,
                                                       parser.combined_dep_placeholder: batch_combined_dep,
                                                       parser.trans_feat_ids: batch_trans_feat_ids,
                                                       parser.trans_feat_sizes: batch_trans_feat_sizes})

                        next_batchstates = [[] for _ in xrange(batch_size)]
                        updated = set()
                        for ik, pred in izip(predsid, p):
                            i, k = ik

                            updated.add(i)

                            if len(batch_states[i][k][1].transitionset()) > 0:
                                # model outputs NLLs so the lower the better
                                sort = sorted(enumerate(pred), key=lambda x: x[1])
                                expanded_beams = 0
                                for choice, score in sort:
                                    newscore = batch_states[i][k][0] - score

                                    if transsys.tuple_trans_from_int(batch_states[i][k][1].transitionset(), choice)[0] in batch_states[i][k][1].transitionset():
                                        candidate = (newscore, batch_states[i][k][1], choice)
                                        if len(next_batchstates[i]) < args.beam_size:
                                            heappush(next_batchstates[i], candidate)
                                        elif newscore > next_batchstates[i][0][0]:
                                            heappushpop(next_batchstates[i], candidate)

                                        expanded_beams += 1
                                        if expanded_beams >= args.beam_size:
                                            break

                        for i in updated:
                            next_batchstates[i] = nlargest(args.beam_size, next_batchstates[i], key=lambda x:x[0])
                            for k, t in enumerate(next_batchstates[i]):
                                score, state, choice = t
                                state = state.clone()
                                transsys.advance(state, choice)
                                next_batchstates[i][k] = (score, state)

                        batch_states = next_batchstates

                        j += 1

                    for i in xrange(batch_size):
                        assert len(batch_finished) == batch_size
                        assert len(batch_finished[i]) > 0, "nothing finished: %d" % (i)
                        assert len(batch_finished[i][0]) > 1, "%s" % (batch_finished[i][0])
                        state_pred = nlargest(1, batch_finished[i], key=lambda x:x[0])[0][1]
                        for t in state_pred.head[1:]:
                            outf.write("%d\t%s\n" % (t[0], invmappings['rel'][t[1]]))
                        outf.write("\n")

                    log.info('Epoch %3d batch %4d' % (epoch, batch))
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluate parsers')

    init_argparse(parser)

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    eval(args)
