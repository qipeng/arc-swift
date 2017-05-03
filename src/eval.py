#!/bin/python
"""A Python re-implementation of CoNLL's eval script, and more.

This script supports evaluation of the output from multiple runs
of the same system, as well as a comparison between two systems.
To obtain the average performance of a system over multiple runs,
simply concatenate the output CoNLL files from the runs and pass
it as a system file. For comparison of two systems, prepare the
output files from both systems, and pass them in as -s and -s2,
respectively. Bootstrap test of differences in the systems'
predictions will be performed, and p-values will be reported with
the null hypothesis that the performance of System 1 (-s) is no
better than that of System 2 (-s2). That is, the smaller this p-
value, the more confident we are in concluding that System 1 is
indeed better than System 2.
"""

import argparse
import logging
import sys
import unicodedata
import multiprocessing
import time
from random import choice
import numpy as np

logging.basicConfig(format="%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='(Simple) Python implementation of attachment score evaluation with significant tests')

parser.add_argument('-g', type=str, required=True, help="Gold file")
parser.add_argument('-s', type=str, required=True, help="System file")
parser.add_argument('-s2', type=str, help="System 2 file")
parser.add_argument('-p', required=False, default=False, action="store_true", help="Evaluate on punctuations as well")
parser.add_argument('-bootstrap', type=int, default=10000, help="Number of bootstraps used to estimate p-value")
parser.add_argument('-nopos', required=False, default=False, action="store_true", help="Exclude POS from evaluation")
parser.add_argument('-language', required=False, default="english", choices=['english', 'conll'], help="If 'conll', evaluate UAS and LAS in accordance with the CONLL eval.pl script (mainly affects treatment of punctuations, if Stanford CoreNLP tokenization is adopted); otherwise laguage-specific punctuation sets will be used")
parser.add_argument('--strict-label', required=False, default=False, action="store_true", help="Use strict relation labels")
parser.add_argument('-fpos', required=False, default=False, action="store_true", help="Use fine-grained POS tags from gold file")

args = parser.parse_args()

# list of unicode punctuations
# adapted from http://stackoverflow.com/a/11066687/1887435
puncts = set([unichr(i) for i in xrange(sys.maxunicode)
              if unicodedata.category(unichr(i)).startswith('P')])

def word_is_punct(word, pos):
    if args.language == 'conll':
        for c in word:
            if c not in puncts:
                return False
        return True
    elif args.language == 'english':
        if args.fpos:
            pos = pos[1] # fpos
        return pos in ['-LRB-', '-RRB-', ',', '.', '``', "''", ':']

def conll2list(filename, include_punct, listgoldpos=None):
    res = []
    res2 = []
    if listgoldpos is None:
        readingpos = True
        listgoldpos = []
    else:
        readingpos = False
    i = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if len(line) <= 0:
                continue

            assert len(line) >= 8, "len(line):%d" % len(line)
            line = line.split('\t')
            word = line[1]
            goldpos = line[3] if not args.fpos else [line[3], line[4]]
            if readingpos:
                listgoldpos += [goldpos]
            else:
                goldpos = listgoldpos[i % len(listgoldpos)]
            pos = line[3] if not args.fpos else [line[3], line[4]]
            head = int(line[6])
            rel = line[7]
            if not args.strict_label:
                rel = rel.split(":")[0]

            is_punct = word_is_punct(word, goldpos)
            if include_punct or not is_punct:
                res += [(word, head, rel)]
            res2 += [(word, pos)]
            i += 1

    return res, res2, listgoldpos

def lists2errs(goldlist, syslist):
    reps = len(syslist) / len(goldlist) # number of systems averaging over

    errhead = [0.0 for _ in goldlist]
    erreither = [0.0 for _ in goldlist]

    rephead = [0.0 for _ in xrange(reps)]
    repeither = [0.0 for _ in xrange(reps)]

    for r in xrange(reps):
        for i in xrange(len(goldlist)):
            j = i + r * len(goldlist)
            assert goldlist[i][0] == syslist[j][0], "Words '%s' and '%s' don't match in gold and sys, please check for validity" % (goldlist[i][0], syslist[j][0])

            if goldlist[i][1] != syslist[j][1]:
                errhead[i] += 1.0 / reps
                erreither[i] += 1.0 / reps

                rephead[r] += 1.0
                repeither[r] += 1.0
            elif goldlist[i][2] != syslist[j][2]:
                erreither[i] += 1.0 / reps

                repeither[r] += 1.0

        rephead[r] /= len(goldlist)
        repeither[r] /= len(goldlist)

    return errhead, erreither, np.std(rephead), np.std(repeither)

def pos_lists2errs(goldlist, syslist):
    reps = len(syslist) / len(goldlist) # number of systems averaging over

    if args.fpos:
        err = [[0.0, 0.0] for _ in goldlist]
        rep = [[0.0, 0.0] for _ in xrange(reps)]
    else:
        err = [0.0 for _ in goldlist]
        rep = [0.0 for _ in xrange(reps)]

    for r in xrange(reps):
        for i in xrange(len(goldlist)):
            j = i + r * len(goldlist)
            assert goldlist[i][0] == syslist[j][0], "Words '%s' and '%s' don't match in gold and sys, please check for validity" % (goldlist[i][0], syslist[j][0])

            if args.fpos:
                for k in range(2):
                    if goldlist[i][1][k] != syslist[j][1][k]:
                        err[i][k] += 1.0 / reps

                        rep[r][k] += 1.0
            else:
                if goldlist[i][1] != syslist[j][1]:
                    err[i] += 1.0 / reps

                    rep[r] += 1.0

        rep[r] /= len(goldlist)

    return err, np.std(rep)

log.info('Reading Gold file...')
gold,gold_,goldpos = conll2list(args.g, args.p)
log.info('Reading System file...')
sys1,sys1_,_ = conll2list(args.s, args.p, goldpos)

assert len(sys1) / len(gold) * len(gold) == len(sys1), "Length of System file is not a multiple of that of Gold file, please check for validity"

errhead1, erreither1, stdhead1, stdeither1 = lists2errs(gold, sys1)

print sum(erreither1)

log.info("Average unlabeled attachment score of System: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errhead1) / len(gold) * 100.0, stdhead1 * 100.0))
log.info("Average labeled attachment score of System: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(erreither1) / len(gold) * 100.0, stdeither1 * 100.0))

if not args.nopos:
    errpos1, stdpos1 = pos_lists2errs(gold_, sys1_)
    if args.fpos:
        errpos1 = map(list, zip(*errpos1))
        stdpos1 = map(list, zip(*stdpos1))
        errfpos1 = errpos1[1]
        stdfpos1 = stdpos1[1]
        errpos1 = errpos1[0]
        stdpos1 = stdpos1[0]
    log.info("Average POS accuracy of System: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errpos1) / len(gold_) * 100.0, stdpos1 * 100.0))
    if args.fpos:
        log.info("Average fPOS accuracy of System: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errfpos1) / len(gold_) * 100.0, stdfpos1 * 100.0))


if args.s2 is not None:
    log.info('Reading System 2 file...')
    if args.s2 == args.s:
        # two input files are the same, no need to actually run evaluation again, nor bootstrap
        log.info("Average unlabeled attachment score of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errhead1) / len(gold) * 100.0, stdhead1 * 100.0))
        log.info("Average labeled attachment score of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(erreither1) / len(gold) * 100.0, stdeither1 * 100.0))
        log.info("P-value for unlabeled attachment score: %f" % (1.0))
        log.info("P-value for labeled attachment score: %f" % (1.0))
        sys.exit(0)

    sys2,sys2_,_ = conll2list(args.s2, args.p, goldpos)

    assert len(sys2) / len(gold) * len(gold) == len(sys2), "Length of System file is not a multiple of that of Gold file, please check for validity"

    errhead2, erreither2, stdhead2, stdeither2 = lists2errs(gold, sys2)
    log.info("Average unlabeled attachment score of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errhead2) / len(gold) * 100.0, stdhead2 * 100.0))
    log.info("Average labeled attachment score of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(erreither2) / len(gold) * 100.0, stdeither2 * 100.0))

    if not args.nopos:
        errpos2, stdpos2 = pos_lists2errs(gold_, sys2_)
        if args.fpos:
            errpos2 = map(list, zip(*errpos2))
            stdpos2 = map(list, zip(*stdpos2))
            errfpos2 = errpos2[1]
            stdfpos2 = stdpos2[1]
            errpos2 = errpos2[0]
            stdpos2 = stdpos2[0]
        log.info("Average POS accuracy of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errpos2) / len(gold_) * 100.0, stdpos2 * 100.0))
        if args.fpos:
            log.info("Average fPOS accuracy of System 2: %8.2f%% (std: %8.4f%%)" % (100.0 - sum(errfpos2) / len(gold_) * 100.0, stdfpos2 * 100.0))

    diffhead = [y - x for x, y in zip(errhead1, errhead2)]
    diffeither = [y - x for x, y in zip(erreither1, erreither2)]

    log.info("Running bootstrap...")
    def bs_one(t):
        diffs, q = t
        bs = [choice(diffs) for _ in diffs]
        sumdiffs = [sum([t[i] for t in diffs]) for i in xrange(len(diffs[0]))]

        res = [1 if sum([t[i] for t in bs]) > 2 * sumdiffs[i] else 0 for i in xrange(len(diffs[0]))]
        q.put(res)

    def bootstrap(diffs, B):
        m = multiprocessing.Manager()
        q = m.Queue()
        pool = multiprocessing.Pool()
        rs = pool.map_async(bs_one, [(diffs, q) for _ in xrange(B)])
        pool.close() # No more work
        while (True):
            if (rs.ready()): break

            log.info('Waiting for %d bootstrap samples to finish...' % (B - q.qsize()))
            time.sleep(1)

        assert(q.qsize() == B), "qsize=%d, B=%d" % (q.qsize(), B)
        count = [0] * len(diffs[0])
        for i in xrange(B):
            qres = q.get()
            for j in xrange(len(diffs[0])):
                count[j] += qres[j]
        assert(q.empty())

        return [(c + 1.0) / (B + 1.0) for c in count]    # smoothed p-value

    if args.nopos:
        diffs = zip(diffhead, diffeither)
    else:
        diffpos = [y - x for x, y in zip(errpos1, errpos2)]
        if args.fpos:
            diff_fpos = [y - x for x, y in zip(errfpos1, errfpos2)]
            diffs = zip(diffhead, diffeither, diff_fpos)
        else:
            diffs = zip(diffhead, diffeither, diffpos)

    p = bootstrap(diffs, args.bootstrap)

    log.info("P-value for unlabeled attachment score: %f" % (p[0]))
    log.info("P-value for labeled attachment score: %f" % (p[1]))

    if not args.nopos:
        log.info("P-value for POS accuracy: %f" % (p[2]))
        if args.fpos:
            log.info("P-value for POS accuracy: %f" % (p[3]))
