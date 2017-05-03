import logging
import os.path as op
from smart_open import smart_open
import cPickle as pickle
from parserstate import ParserState
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid
import numpy as np

transition_dims = ['action', 'n', 'rel', 'pos', 'fpos']
transition_pos = {v:i for i, v in enumerate(transition_dims)}
floatX = np.float32

def transsys_lookup(k):
    lookup = {"ASw": ArcSwift,
              "AER": ArcEagerReduce,
              "AES": ArcEagerShift,
              "ASd": ArcStandard,
              "AH" : ArcHybrid,}
    return lookup[k]

def process_example(conll_lines, seq_lines, vocab, mappings, transsys, fpos=False, log=None):
    if fpos:
        res = [[] for _ in xrange(4)]
    else:
        res = [[] for _ in xrange(3)]
    res[0] = [vocab[u'<ROOT>']] + [vocab[u'<UNK>'] if line.split()[1] not in vocab else vocab[line.split()[1]] for line in conll_lines]
    for line in seq_lines:
        line = line.split()
        try:
            fields = transsys.trans_from_line(line)
        except ValueError as e:
            log.error('Encountered unknown transition type "%s" in sequences file, ignoring...' % (str(e)))
            return None

        vector_form = []
        for k in transition_dims:
            if k in fields:
                if k in mappings:
                    fields[k] = mappings[k][fields[k]]
                vector_form += [fields[k]]
            else:
                vector_form += [-1] # this should never be used

        res[1] += [vector_form]

    # gold POS
    res[2] = [len(mappings['pos'])] + [mappings['pos'][line.split()[3]] for line in conll_lines]
    if fpos:
        # fine-grained POS
        res[3] = [len(mappings['fpos'])] + [mappings['fpos'][line.split()[4]] for line in conll_lines]

    return tuple(res)

def read_data(conll_file, seq_file, vocab, mappings, transsys, fpos=False, log=None):
    log.info('Reading dependency parse data...')
    max_sent_len = -1
    max_seq_len = -1
    res = []
    with smart_open(conll_file, 'r') as conllf:
        if seq_file is not None:
            with smart_open(seq_file, 'r') as seqf:
                lines1 = []
                lines2 = []

                while True:
                    line1 = conllf.readline().decode('utf-8')

                    while line1:
                        line1_ = line1.strip()

                        if len(line1_) <= 0:
                            break

                        lines1 += [line1_]
                        line1 = conllf.readline().decode('utf-8')

                    line2 = seqf.readline().decode('utf-8')

                    while line2:
                        line2_ = line2.strip()

                        if len(line2_) <= 0:
                            break

                        lines2 += [line2_]
                        line2 = seqf.readline().decode('utf-8')

                    if not line1 or not line2:
                        break

                    if len(lines1) + 1 > max_sent_len:
                        max_sent_len = len(lines1) + 1
                    if len(lines2) > max_seq_len:
                        max_seq_len = len(lines2)
                    t = process_example(lines1, lines2, vocab, mappings, transsys, fpos)
                    if t:
                        res += [t]

                    lines1 = []
                    lines2 = []
        else:
            lines1 = []
            while True:
                line1 = conllf.readline().decode('utf-8')

                while line1:
                    line1_ = line1.strip()

                    if len(line1_) <= 0:
                        break

                    lines1 += [line1_]
                    line1 = conllf.readline().decode('utf-8')

                if not line1:
                    break

                if len(lines1) + 1 > max_sent_len:
                    max_sent_len = len(lines1) + 1
                t = process_example(lines1, [], vocab, mappings, transsys, fpos)
                if t:
                    res += [t]

                lines1 = []

    sent_length = max_sent_len
    log.info('%d examples read, max sentence length %d, max transition count %d.' % (len(res), sent_length, max_seq_len))
    return res, sent_length, max_seq_len

def read_vocab(conll_file, wordvec_file, vocab_file, wordvec_dim, min_count=3, log=None):
    log.info('Reading vocabulary...')

    if vocab_file is not None and op.exists(vocab_file):
        with smart_open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            vecs = pickle.load(f)
            i1 = pickle.load(f)

        log.info('Read %d words from saved vocabulary file "%s".' % (len(vocab), vocab_file))

        return vocab, vecs, i1

    vocab = dict()
    vecs = []
    i = i1 = 0
    if wordvec_file and op.exists(wordvec_file):
        with smart_open(wordvec_file, 'r') as f:
            for line in f:
                try:
                    line = line.decode('utf-8').strip().split()
                except UnicodeDecodeError:
                    continue
                word = " ".join(line[0:(len(line) - wordvec_dim)])
                if word not in vocab:
                    vocab[word] = i
                    i += 1

                    if i % 10000 == 0:
                        log.info('Read %d words from the word vector file' % (i))

                    if wordvec_dim is not None and wordvec_dim != len(line) - len(word.split()):
                        log.warn('Specified word embedding dim(%d) is different from that in the word embeddings file(%d) for word "%s", skipping...' % (wordvec_dim, len(line)-len(word.split()), word))
                    elif wordvec_dim is None:
                        wordvec_dim = len(line) - 1

                    vecs += [np.array([float(x) for x in line[len(word.split()):]], dtype=floatX)]

        i1 = i
        log.debug('%d words read from word vector file' % i)

    i0 = i1

    count = dict()
    with smart_open(conll_file, 'r') as f:
        totalcount = 0
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue

            line = line.split()
            if line[1] in count:
                count[line[1]] += 1
            else:
                count[line[1]] = 1

                if line[1] not in vocab:
                    totalcount += 1

                    if totalcount % 10000 == 0:
                        log.info('Read %d words from the CoNLL file' % (totalcount))

        pruned = 0
        for w in count:
            if count[w] < min_count:
                pruned += 1
                continue

            if w not in vocab and w.lower() in vocab:
                vecs += [vecs[vocab[w.lower()]]]
                i1 += 1
                vocab[w] = i
                i += 1

        for w in count:
            if count[w] < min_count:
                continue

            if w not in vocab and w.lower() not in vocab:
                vecs += [np.random.randn(wordvec_dim).astype(floatX)]
                vocab[w] = i
                i += 1

        for w in ['<UNK>', '<ROOT>']:
            if w not in vocab:
                vocab[w] = i
                i += 1

            vecs += [np.random.randn(wordvec_dim).astype(floatX)]

        log.debug('%d words retained from CoNLL file, %d initialized with word vectors, %d low frequency words pruned' % (i-i0, i1-i0, pruned))

    log.info('%d words read.' % (i))

    log.info('Normalizing word embeddings...')
    vecs = np.array(vecs, dtype=floatX)

    # re-center and normalize word vectors
    if i1 > 0:
        mu = np.mean(vecs[:i0], axis = 0)
        sigma = np.std(vecs[:i0], axis = 0)
        vecs[:i1] = (vecs[:i1] - mu.reshape(1, -1)) / sigma.reshape(1, -1)

    if vocab_file is not None:
        log.info('Saving vocab and normalized embeddings to "%s"...' % (vocab_file))
        with smart_open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(vecs, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(i1, f, pickle.HIGHEST_PROTOCOL)

        log.info('Done.')

    return vocab, vecs, i1

def read_mappings(mappings_file, transsys, log=None):
    i = 0
    res = dict()
    res2 = dict()
    with smart_open(mappings_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("::"):
                currentkey = line[2:]
                res[currentkey] = dict()
                res2[currentkey] = []
                i = 0
            else:
                res[currentkey][line] = i
                res2[currentkey] += [line]
                i += 1

    res['action'] = {k: i for i, k in enumerate(transsys.actions_list())}
    res2['action'] = transsys.actions_list()

    return res, res2

def featurize_state(state, mappings, t=None):
    buf = state.buf
    stack = state.stack
    head = state.head

    ACTION = transition_pos['action']
    N = transition_pos['n']
    REL = transition_pos['rel']
    POS = transition_pos['pos']

    SHIFT = mappings['action']['Shift']
    LEFTARC = mappings['action']['Left-Arc']
    RIGHTARC = mappings['action']['Right-Arc']
    RELS = len(mappings['rel'])

    label = -1
    featdim = 0

    possible_trans = state.transitionset()
    transsys = state.transsys

    assert t is None or len(possible_trans) > 0
    assert transsys is not None
    if isinstance(transsys, ArcSwift):
        feat = [None for _ in possible_trans]
        assert(t is None or len(mappings['pos']) > t[POS] >= 0 or t[ACTION] == LEFTARC)
        pos = -1 if t is None or t[ACTION] in [LEFTARC] else t[POS]

        for i, t1 in enumerate(possible_trans):
            if t1[0] == SHIFT:
                feat[i] = (SHIFT, -1, -1, buf[0], pos)
                if t is not None and t[ACTION] == SHIFT:
                    label = featdim
                featdim += 1
            elif t1[0] == LEFTARC or t1[0] == RIGHTARC:
                si = t1[1]
                head = buf[0] if t1[0] == LEFTARC else stack[si]
                dep = stack[si] if t1[0] == LEFTARC else buf[0]
                feat[i] = (t1[0], head, dep, buf[0], pos)
                if t is not None and t[ACTION] == t1[0] and t[N] == si+1:
                    label = featdim + t[REL]
                featdim += RELS

        if t is not None:
            if possible_trans[0][0] == SHIFT:
                assert(0 <= (label-1)/RELS+1 < len(feat))
            else:
                assert(0 <= label/RELS < len(feat))
            assert label == transsys.tuple_trans_to_int(possible_trans, (t[ACTION], t[N]-1, t[REL])), "action: (%d, %d, %d), mine: %d, theirs: %d, cand:%s" % (t[ACTION], t[N]-1, t[REL], label, transsys.tuple_trans_to_int(possible_trans, (t[ACTION], t[N]-1, t[REL])), possible_trans)

        assert(len(feat) == len(possible_trans))
    elif isinstance(transsys, ArcEagerReduce) or isinstance(transsys, ArcEagerShift):
        if len(possible_trans) == 0:
            return []
        REDUCE = mappings['action']['Reduce']
        assert(t is None or len(mappings['pos']) > t[POS] >= 0 or (t[ACTION] == LEFTARC or t[ACTION] == REDUCE))
        assert(t is None or (t[ACTION],) in possible_trans)
        pos = -1 if t is None or t[ACTION] in [LEFTARC, REDUCE] else t[POS]

        stack_top3 = [-1] * 3
        stack_top3[:min(len(stack), 3)] = stack[:min(len(stack), 3)]
        feat = tuple(stack_top3 + [buf[0]] + [pos])

        for t1 in [SHIFT, REDUCE, LEFTARC, RIGHTARC]:
            if t1 == SHIFT or t1 == REDUCE:
                if t is not None and t[ACTION] == t1:
                    label = featdim
                featdim += 1
            elif t1 == LEFTARC or t1 == RIGHTARC:
                if t is not None and t[ACTION] == t1:
                    label = featdim + t[REL]
                featdim += RELS

        assert t is None or label == transsys.tuple_trans_to_int(possible_trans, (t[ACTION], t[REL]))
    elif isinstance(transsys, ArcStandard) or isinstance(transsys, ArcHybrid):
        if len(possible_trans) == 0:
            return []
        assert(t is None or len(mappings['pos']) > t[POS] >= 0 or (t[ACTION] == LEFTARC or t[ACTION] == RIGHTARC))
        assert(t is None or (t[ACTION],) in possible_trans)
        pos = -1 if t is None or t[ACTION] != SHIFT else t[POS]

        stack_top3 = [-1] * 3
        stack_top3[:min(len(stack), 3)] = stack[:min(len(stack), 3)]
        feat = tuple(stack_top3 + [-1 if len(buf) == 0 else buf[0]] + [pos])

        for t1 in [SHIFT, LEFTARC, RIGHTARC]:
            if t1 == SHIFT:
                if t is not None and t[ACTION] == t1:
                    label = featdim
                featdim += 1
            elif t1 == LEFTARC or t1 == RIGHTARC:
                if t is not None and t[ACTION] == t1:
                    label = featdim + t[REL]
                featdim += RELS

        assert t is None or label == transsys.tuple_trans_to_int(possible_trans, (t[ACTION], t[REL]))

    assert(t is None or label >= 0), "transition: %s, label: %d, featdim: %d, possible_trans: %s" % (str(t), label, featdim, str(possible_trans))

    if t is None:
        return feat
    return feat, label

def featurize_transitions(data, mappings, invmappings, feat_file, transsys, log=None):
    if feat_file is not None and op.exists(feat_file):
        with smart_open(feat_file, 'rb') as f:
            res = pickle.load(f)
            max_feat_size = pickle.load(f)

        log.info('Read %d featurized examples from saved feature file "%s".' % (len(res), feat_file))

        return res, max_feat_size

    log.info('Featurizing %d examples...' % (len(data)))
    max_feat_size = -1
    res = []

    count = 0
    transsys = transsys(mappings, invmappings)

    for t in data:
        sent, trans = t[:2]
        state = ParserState(sent, transsys=transsys)

        feats = []
        labels = []
        featsizes = []
        for t in trans:
            feat, label = featurize_state(state, mappings, t)
            transsys.advance(state, label)

            max_feat_size = max(max_feat_size, len(feat))
            feats += [feat]
            labels += [label]
            featsizes += [len(feat)]

        assert(len(feats) == len(labels))
        res += [[feats, featsizes, labels]]

        count += 1
        if (count) % 100 == 0:
            log.debug("Featurized %d examples..." % (count))

    assert(len(res) == len(data))
    log.info("%d examples featurized, maximum feature size=%d" % (len(res), (max_feat_size-1)*len(mappings['rel'])+1))

    if feat_file is not None:
        log.info('Saving %d featurized examples to feature file "%s"...' % (len(res), feat_file))
        with smart_open(feat_file, 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(max_feat_size, f, pickle.HIGHEST_PROTOCOL)

        log.info('Done.')

    return res, max_feat_size

def read_gold_parserstates(fin, transsys, fpos=False):
    def processlines(lines):
        arcs = [dict() for i in range(len(lines)+1)]

        pos = ["" for i in xrange(len(lines)+1)]
        fpos = ["" for i in xrange(len(lines)+1)]

        for i, line in enumerate(lines):
            pos[i+1] = line[3] # fine-grained
            fpos[i+1] = line[4]
            parent = int(line[6])
            relation = line[7]
            arcs[parent][i+1] = transsys.mappings['rel'][relation]

        res = [ParserState(["<ROOT>"] + lines, transsys=transsys, goldrels=arcs), pos]
        if fpos:
            res += [fpos]
        else:
            res == [None]
        return res
    res = []

    lines = []
    line = fin.readline().decode('utf-8')
    while line:
        line = line.strip().split()

        if len(line) == 0:
            res += [processlines(lines)]
            lines = []
        else:
            lines += [line]

        line = fin.readline().decode('utf-8')

    if len(lines) > 0:
        res += [processlines(lines)]

    return res

def write_gold_trans(tpl, fout):
    state, pos, fpos = tpl
    transsys = state.transsys
    while len(state.transitionset()) > 0:
        t = transsys.goldtransition(state)

        fout.write("%s\n" % transsys.trans_to_str(t, state, pos, fpos))

        transsys.advance(state, t)

    fout.write("\n")

def multi_argmin(lst):
    minval = 1e10
    res = []
    for i, v in enumerate(lst):
        if v < minval:
            minval = v
            res = [i]
        elif v == minval:
            res += [i]

    return res
