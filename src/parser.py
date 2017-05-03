import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn#, rnn_cell
from tensorflow.contrib import rnn as rnn_cell

from utils import *
from layers import DenseLayer, MergeLayer

class Parser:
    def __init__(self, args, vecs, pretrained, mappings, invmappings, sent_length, trans_length, feat_dim, log, train=True):
        self.train = train
        self.args = args

        self.sent_length = sent_length

        self.init_comp_graph(args, vecs, pretrained, mappings, invmappings, trans_length, feat_dim, log)

    def init_comp_graph(self, args, vecs, pretrained, mappings, invmappings, trans_length, feat_dim, log):
        keep_prob = args.keep_prob if self.train else 1.0

        feat_shape = [5] if args.transsys != 'ASw' else ([feat_dim, 5] if self.train else [self.sent_length, 5])

        # build computational graph
        log.info('Building computational graph, this might take a while...')
        # POS BiLSTM
        log.debug('Building computational graph for the POS-BiLSTM...')
        word_emb_dim = vecs.shape[1]
        # Uppercased words are initialized with lowercased vectors, but finetuned separately
        pretrained_base = tf.Variable(vecs[:pretrained], trainable=False)
        pretrained_delta = tf.Variable(np.zeros(vecs[:pretrained].shape, dtype=floatX))
        pretrained_emb = tf.add(pretrained_base, pretrained_delta)
        random_emb = tf.Variable(vecs[pretrained:])
        embeddings = tf.concat([pretrained_emb, random_emb], 0)
        self.words = tf.placeholder(tf.int32, [args.batch_size, self.sent_length])
        self.words2 = tf.placeholder(tf.int32, [args.batch_size, self.sent_length])
        self.sent_lengths = tf.placeholder(tf.int32, [args.batch_size])
        word_emb = tf.nn.embedding_lookup(embeddings, self.words)
        word_emb2 = tf.nn.embedding_lookup(embeddings, self.words2)

        with tf.variable_scope('bilstm1'):
            lstm_fw = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True) for _ in range(args.layers)], state_is_tuple=True)
            lstm_bw = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True) for _ in range(args.layers)], state_is_tuple=True)

            bilstm_outputs, _ = rnn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, word_emb, sequence_length=self.sent_lengths, dtype=tf.float32)


        # POS
        self.gold_pos = tf.placeholder(tf.int32, [args.batch_size, self.sent_length])
        if args.fpos:
            self.gold_fpos = tf.placeholder(tf.int32, [args.batch_size, self.sent_length])

        # POS system
        log.debug('Building computational graph for the POS-tagging system...')
        if not args.no_pos:
            log.debug('Building computational graph for dense layers following Parse-BiLSTM...')
            pos_densesizes = [args.hidden_size] + [int(x) for x in args.pos_dense_layers.split(',')] + [args.pos_emb_dim, len(mappings['pos'])]
            pos_denselayers = len(pos_densesizes) - 1

            pos_dense_inputs = [tf.reshape(x, [-1, args.hidden_size]) for x in bilstm_outputs]
            pos_dense = [MergeLayer(pos_densesizes[0], pos_densesizes[0], pos_densesizes[1], keepProb=keep_prob, combination='affine')]
            pos_dense += [DenseLayer(pos_densesizes[i], pos_densesizes[i+1], keepProb=keep_prob) for i in xrange(1, pos_denselayers - 1)]
            # split representations for head and dependent
            pos_dense += [DenseLayer(pos_densesizes[-2], pos_densesizes[-1], keepProb=keep_prob, nl=lambda x:x)]

            pos_dense_intermediate = pos_dense[0](pos_dense_inputs[0], pos_dense_inputs[1])

            for l in xrange(1, pos_denselayers-1):
                pos_dense_intermediate = pos_dense[l](pos_dense_intermediate)

            pos_dense_outputs = tf.reshape(pos_dense[-1](pos_dense_intermediate), [args.batch_size, -1, len(mappings['pos'])])
            if args.fpos:
                fpos_dense = DenseLayer(args.pos_emb_dim, len(mappings['fpos']), keepProb=keep_prob, nl=lambda x:x)
                fpos_dense_outputs = tf.reshape(fpos_dense(pos_dense_intermediate), [args.batch_size, -1, len(mappings['fpos'])])
        else:
            pos_dense_outputs = [None for _ in range(args.batch_size)]

        pos_trainables = tf.Variable(tf.truncated_normal((len(mappings['pos']), args.pos_emb_dim)),
                dtype=tf.float32, name='pos_trainables')
        pos_untrainable = tf.Variable(tf.zeros((1, args.pos_emb_dim), dtype=tf.float32), trainable=False)
        pos_embeddings = tf.concat([pos_trainables, pos_untrainable], 0)

        pos_loss_pred_ = lambda i: self.pos_loss_pred(i, pos_embeddings, pos_dense_outputs[i], len(mappings['pos']), self.gold_pos, pos_trainables)

        if self.train:
            pos_losses = tf.multiply(args.pos_mult, tf.map_fn(lambda i: pos_loss_pred_(i)[0], tf.range(args.batch_size), parallel_iterations=args.batch_size, dtype=tf.float32))
        else:
            self.pos_preds = tf.map_fn(lambda i: pos_loss_pred_(i)[0], tf.range(args.batch_size), parallel_iterations=args.batch_size)

        self.pos_embs = tf.map_fn(lambda i: pos_loss_pred_(i)[1], tf.range(args.batch_size), parallel_iterations=args.batch_size, dtype=tf.float32)

        if args.fpos:
            fpos_trainables = tf.Variable(tf.truncated_normal((len(mappings['fpos']), args.pos_emb_dim)),
                    dtype=tf.float32, name='fpos_trainables')
            fpos_untrainable = tf.Variable(tf.zeros((1, args.pos_emb_dim), dtype=tf.float32), trainable=False)
            fpos_embeddings = tf.concat([fpos_trainables, fpos_untrainable], 0)

            fpos_loss_pred_ = lambda i: self.pos_loss_pred(i, fpos_embeddings, fpos_dense_outputs[i], len(mappings['fpos']), self.gold_fpos, fpos_trainables)

            if self.train:
                fpos_losses = tf.multiply(args.pos_mult, tf.map_fn(lambda i: fpos_loss_pred_(i)[0], tf.range(args.batch_size), parallel_iterations=args.batch_size, dtype=tf.float32))
                pos_losses = pos_losses + fpos_losses
            else:
                self.fpos_preds = tf.map_fn(lambda i: fpos_loss_pred_(i)[0], tf.range(args.batch_size), parallel_iterations=args.batch_size)

            self.fpos_embs = tf.map_fn(lambda i: fpos_loss_pred_(i)[1], tf.range(args.batch_size), parallel_iterations=args.batch_size, dtype=tf.float32)

        bilstm_outputs = tf.concat([bilstm_outputs[0], bilstm_outputs[1]], 2)

        # Concatenate tagger BiLSTM outputs as Parser BiLSTM input
        concat_list = [bilstm_outputs]
        dim = args.hidden_size * 2

        concat_list += [self.pos_embs]
        dim += args.pos_emb_dim

        if args.fpos:
            concat_list += [self.fpos_embs]
            dim += args.pos_emb_dim

        bilstm2_inputs = tf.reshape(tf.concat(concat_list, 2), [args.batch_size, -1, dim])

        # Parse BiLSTM
        log.debug('Building computational graph for the Parse-BiLSTM...')

        with tf.variable_scope('bilstm2'):
            lstm2_fw = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True) for _ in range(args.layers2)], state_is_tuple=True)
            lstm2_bw = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(args.hidden_size, state_is_tuple=True) for _ in range(args.layers2)], state_is_tuple=True)

            bilstm2_outputs, _ = rnn.bidirectional_dynamic_rnn(lstm2_fw, lstm2_bw, bilstm2_inputs, sequence_length=self.sent_lengths, dtype=tf.float32)

        # Dense layer(s)
        log.debug('Building computational graph for dense layers following Parse-BiLSTM...')
        densesizes = [args.hidden_size] + [int(x) for x in args.dense_layers.split(',')] + [args.rel_emb_dim]
        denselayers = len(densesizes) - 1

        dense_inputs = [tf.reshape(x, [-1, args.hidden_size]) for x in bilstm2_outputs]
        if denselayers == 1:
            dense = [[MergeLayer(densesizes[0], densesizes[0], densesizes[1], keepProb=keep_prob, combination='affine') for _ in xrange(2)]]
            dense_outputs = [dense[0][j](dense_inputs[0], dense_inputs[1]) for j in xrange(2)]
        else:
            dense = [MergeLayer(densesizes[0], densesizes[0], densesizes[1], keepProb=keep_prob, combination='affine')]
            dense += [DenseLayer(densesizes[i], densesizes[i+1], keepProb=keep_prob) for i in xrange(1, denselayers - 1)]
            # split representations for head and dependent
            dense += [[DenseLayer(densesizes[-2], densesizes[-1], keepProb=keep_prob) for _ in xrange(2)]]

            dense_outputs = dense[0](dense_inputs[0], dense_inputs[1])

            for l in xrange(1, denselayers-1):
                dense_outputs = dense[l](dense_outputs)

            dense_outputs = [dense[-1][j](dense_outputs) for j in xrange(2)]

        dense_outputs = [tf.reshape(x, [args.batch_size, -1, args.rel_emb_dim]) for x in dense_outputs]

        self.combined_head = dense_outputs[0]
        self.combined_dep  = dense_outputs[1]

        # transition system
        log.debug('Building computational graph for the transition system...')
        if self.train:
            self.trans_feat_ids = tf.placeholder(tf.int32, [args.batch_size, trans_length] + feat_shape)
            self.trans_feat_sizes = tf.placeholder(tf.int32, [args.batch_size, trans_length])
            self.trans_labels = tf.placeholder(tf.int32, [args.batch_size, trans_length])
            self.trans_lengths = tf.placeholder(tf.int32, [args.batch_size])
        else:
            self.trans_feat_ids = tf.placeholder(tf.int32, [None] + feat_shape)
            self.trans_feat_sizes = tf.placeholder(tf.int32, [None])

        self.rel_merge = MergeLayer(args.rel_emb_dim, args.rel_emb_dim, args.rel_emb_dim,
                keepProb=keep_prob, combination=args.combination)

        if args.transsys == 'ASw':
            self.rel_dense = DenseLayer(args.rel_emb_dim, len(mappings['rel']), nl=lambda x:x)

            transition_dense = MergeLayer(args.rel_emb_dim, args.rel_emb_dim, 1, nl=lambda x:x, combination=args.combination)
            self.transition_logit = transition_dense(tf.reshape(self.combined_head, [-1, args.rel_emb_dim]),
                    tf.reshape(self.combined_dep, [-1, args.rel_emb_dim]))
            self.transition_logit = tf.reshape(self.transition_logit, (args.batch_size, -1))
        elif args.transsys in ['AER', 'AES']:
            self.rel_dense = DenseLayer(args.rel_emb_dim * 4, 2 + 2 * len(mappings['rel']), nl=lambda x:x)
        elif args.transsys in ['ASd', 'AH']:
            self.rel_dense = DenseLayer(args.rel_emb_dim * 4, 1 + 2 * len(mappings['rel']), nl=lambda x:x)

        SHIFT = mappings['action']['Shift']

        if self.train:
            if args.transsys == 'ASw':
                trans_loss_f = lambda i, j: self.ASw_transition_loss_pred(i, j, self.combined_head[i], self.combined_dep[i], self.transition_logit[i], SHIFT)
            else:
                trans_loss_f = lambda i, j: self.traditional_transition_loss_pred(i, j, self.combined_head[i], self.combined_dep[i])

            def _ex_loss(i):
                trans_loss = tf.reduce_sum(tf.map_fn(lambda j: trans_loss_f(i, j), tf.range(self.trans_lengths[i]), dtype=tf.float32, parallel_iterations=100))
                if not args.no_pos:
                    loss = tf.add(pos_losses[i], trans_loss)
                else:
                    loss = trans_loss

                return loss

            losses = tf.map_fn(_ex_loss, tf.range(args.batch_size), dtype=tf.float32, parallel_iterations=100)

            self._loss = tf.reduce_mean(losses)
        else:
            self.combined_head_placeholder = tf.placeholder(tf.float32, (None, self.sent_length, args.rel_emb_dim))
            self.combined_dep_placeholder  = tf.placeholder(tf.float32, (None, self.sent_length, args.rel_emb_dim))
            if args.transsys == 'ASw':
                self.trans_logit_placeholder = tf.placeholder(tf.float32, (None, self.sent_length))
                trans_pred = lambda i, k: self.ASw_transition_loss_pred(i, k, self.combined_head_placeholder[i], self.combined_dep_placeholder[i], self.trans_logit_placeholder[i], SHIFT)
                self.pred_output_size = self.sent_length * len(mappings['rel']) + 1
            else:
                trans_pred = lambda i, k: self.traditional_transition_loss_pred(i, k, self.combined_head_placeholder[i], self.combined_dep_placeholder[i])
                if args.transsys in ['AES', 'AER']:
                    self.pred_output_size = 2 * len(mappings['rel']) + 2
                elif args.transsys in ['ASd', 'AH']:
                    self.pred_output_size = 2 * len(mappings['rel']) + 1

            self._trans_predictors = [[trans_pred(i,k) for k in range(args.beam_size)] for i in xrange(args.batch_size)]

    @property
    def loss(self):
        return self._loss

    @property
    def trans_predictors(self):
        return self._trans_predictors

    def traditional_transition_loss_pred(self, i, j, combined_head, combined_dep):
        rel_trans_feat_ids = self.trans_feat_ids[i*self.args.beam_size+j] if not self.train else self.trans_feat_ids[i, j]
        rel_head = tf.reshape(tf.gather(combined_head, rel_trans_feat_ids[:4]), [4, self.args.rel_emb_dim])
        rel_dep  = tf.reshape(tf.gather(combined_dep,  rel_trans_feat_ids[:4]), [4, self.args.rel_emb_dim])

        mask = tf.cast(tf.reshape(tf.greater_equal(rel_trans_feat_ids[:4], 0), [4,1]), tf.float32)
        rel_head = tf.multiply(mask, rel_head)
        rel_dep = tf.multiply(mask, rel_dep)

        rel_hid = self.rel_merge(rel_head, rel_dep)
        rel_logit = self.rel_dense(tf.reshape(rel_hid, [1, -1]))
        rel_logit = tf.reshape(rel_logit, [-1])

        log_partition = tf.reduce_logsumexp(rel_logit)
        if self.train:
            res = log_partition - rel_logit[self.trans_labels[i, j]]

            return res
        else:
            arc_pred = log_partition - rel_logit

            return arc_pred

    def ASw_transition_loss_pred(self, i, j, combined_head, combined_dep, transition_logit, SHIFT):
        # extract relevant portions of params
        rel_trans_feat_ids = self.trans_feat_ids[i*self.args.beam_size+j] if not self.train else self.trans_feat_ids[i, j]
        rel_trans_feat_size = self.trans_feat_sizes[i*self.args.beam_size+j] if not self.train else self.trans_feat_sizes[i, j]

        # core computations
        has_shift = tf.cond(tf.equal(rel_trans_feat_ids[0, 0], SHIFT), lambda: tf.constant(1), lambda: tf.constant(0))
        arc_trans_count = rel_trans_feat_size - has_shift

        arc_trans_feat_ids = tf.gather(rel_trans_feat_ids, tf.range(has_shift, rel_trans_feat_size))
        rel_head = tf.reshape(tf.gather(combined_head, arc_trans_feat_ids[:, 1]), [arc_trans_count, self.args.rel_emb_dim])
        rel_dep  = tf.reshape(tf.gather(combined_dep,  arc_trans_feat_ids[:, 2]), [arc_trans_count, self.args.rel_emb_dim])

        rel_hid = self.rel_merge(rel_head, rel_dep)
        rel_logit = self.rel_dense(rel_hid)
        arc_logit = tf.reshape(rel_logit, [-1])

        def logaddexp(a, b):
            mx = tf.maximum(a, b)
            return tf.log(tf.exp(a-mx) + tf.exp(b-mx)) + mx

        if self.train:
            # compute a loss and return it
            log_partition = tf.reduce_logsumexp(arc_logit)
            log_partition = tf.cond(tf.greater(has_shift, 0),
                    lambda: logaddexp(log_partition, transition_logit[rel_trans_feat_ids[0, 3]]),
                    lambda: log_partition)
            arc_logit = log_partition - arc_logit

            res = tf.cond(tf.greater(has_shift, 0),
                        lambda: tf.cond(tf.greater(self.trans_labels[i, j], 0),
                            lambda: arc_logit[self.trans_labels[i, j]-1],
                            lambda: log_partition - transition_logit[rel_trans_feat_ids[0, 3]]),
                        lambda: arc_logit[self.trans_labels[i, j]])

            return res
        else:
            # just return predictions
            arc_logit = tf.reshape(rel_logit, [-1])
            log_partition = tf.reduce_logsumexp(arc_logit)
            log_partition = tf.cond(tf.greater(has_shift, 0),
                    lambda: logaddexp(log_partition, transition_logit[rel_trans_feat_ids[0, 3]]),
                    lambda: log_partition)
            arc_logit = log_partition - arc_logit

            arc_pred = tf.cond(tf.greater(has_shift, 0),
                lambda: tf.concat([tf.reshape(log_partition - transition_logit[rel_trans_feat_ids[0, 3]], (-1,1)),
                         tf.reshape(arc_logit, (-1,1))], 0),
                lambda: tf.reshape(arc_logit, (-1, 1)))

            # correct shape
            current_output_shape = has_shift + arc_trans_count * rel_logit.get_shape()[1]
            arc_pred = tf.concat([arc_pred, 1e20 * tf.ones((tf.subtract(self.pred_output_size, current_output_shape), 1), dtype=tf.float32)], 0)
            arc_pred = tf.reshape(arc_pred, [-1])

            return arc_pred

    def pos_loss_pred(self, i, pos_embeddings, pos_logit, NUM_POS, gold_pos, pos_trainables):
        if self.args.no_pos:
            pos_emb = tf.nn.embedding_lookup(pos_embeddings, gold_pos[i])
            if self.train:
                return 0, pos_emb
            else:
                return tf.gather(gold_pos[i], tf.range(1, self.sent_length)), pos_emb
        else:
            pos_logit = pos_logit[1:]

            log_partition = tf.reduce_logsumexp(pos_logit, [1])

            pos_pred = tf.exp(pos_logit - tf.reshape(log_partition, (-1, 1)))
            pos_emb = tf.concat([tf.reshape(tf.nn.embedding_lookup(pos_embeddings, NUM_POS), (1, -1)),
                tf.matmul(pos_pred, pos_trainables)], 0)

            if self.train:
                loss = tf.reduce_sum(tf.gather(log_partition, tf.range(self.sent_lengths[i]-1))
                    - tf.gather(tf.reshape(pos_logit, [-1]),
                        tf.range(self.sent_lengths[i]-1) * NUM_POS
                        + tf.gather(gold_pos[i], tf.range(1, self.sent_lengths[i]))))

                return loss, pos_emb
            else:
                return tf.cast(tf.argmax(pos_pred, 1), tf.int32), pos_emb
