"""
This script aims at verifying the implementation of transition
systems with the most straightforward code possible. This script
shouldn't be run manually, but should instead be invoked by
test_oracle.sh.
"""

import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('filein', type=str, help="Input CONLL file")
parser.add_argument('fileout', type=str, help="Output oracle sequence file")
parser.add_argument('--transsys', type=str, choices=['ASw', 'AER', 'AES', 'ASd', 'AH'], help="Transition system to use", default="ASw")

args = parser.parse_args()

"""
This function blindly follows the given oracle sequence with an
implementation as obvious as possible, without any considerations
about pre- or post-conditions. The goal is to verify that the
transition systems are generating sequences that achieve the gold
parse, as well as being more interpretable by humans.
"""
def processlines(lines, fout):
    stack = [0]
    Nwords = 0
    for t in lines:
        if args.transsys in ['ASw', 'AER', 'AES']:
            if t[0] == 'Shift' or t[0] == 'Right-Arc':
                Nwords += 1
        elif args.transsys in ['ASd', 'AH']:
            if t[0] ==  'Shift':
                Nwords += 1

    buf = [i+1 for i in xrange(Nwords)]

    parent = [-1 for i in xrange(Nwords + 1)]
    output = ["" for i in xrange(Nwords + 1)]

    pos = ["" for i in xrange(Nwords + 1)]

    for t in lines:
        j = None if len(buf) == 0 else buf[0]
        if args.transsys == 'ASw':
            if t[0] == 'Left-Arc':
                n = int(t[1]) - 1
                relation = t[2]
                parent[stack[n]] = j
                output[stack[n]] = "%d\t%s" % (j, relation)
                stack = stack[(n+1):]
            elif t[0] == 'Right-Arc':
                pos[j] = t[3]
                n = int(t[1]) - 1
                relation = t[2]
                parent[j] = stack[n]
                output[j] = "%d\t%s" % (stack[n], relation)
                buf = buf[1:]
                stack = [j] + stack[n:]
            else:
                pos[j] = t[1]
                stack = [j] + stack
                buf = buf[1:]
        elif args.transsys in ['AES', 'AER']:
            if t[0] == 'Left-Arc':
                relation = t[1]
                parent[stack[0]] = j
                output[stack[0]] = "%d\t%s" % (j, relation)
                stack = stack[1:]
            elif t[0] == 'Right-Arc':
                pos[j] = t[2]
                relation = t[1]
                parent[j] = stack[0]
                output[j] = "%d\t%s" % (stack[0], relation)
                buf = buf[1:]
                stack = [j] + stack
            elif t[0] == 'Shift':
                pos[j] = t[1]
                stack = [j] + stack
                buf = buf[1:]
            else:
                stack = stack[1:]
        elif args.transsys in ['ASd']:
            if t[0] == 'Left-Arc':
                relation = t[1]
                parent[stack[1]] = stack[0]
                output[stack[1]] = "%d\t%s" % (stack[0], relation)
                stack = [stack[0]] + stack[2:]
            elif t[0] == 'Right-Arc':
                relation = t[1]
                parent[stack[0]] = stack[1]
                output[stack[0]] = "%d\t%s" % (stack[1], relation)
                stack = stack[1:]
            elif t[0] == 'Shift':
                pos[j] = t[1]
                stack = [j] + stack
                buf = buf[1:]
        elif args.transsys in ['AH']:
            if t[0] == 'Left-Arc':
                relation = t[1]
                parent[stack[0]] = j
                output[stack[0]] = "%d\t%s" % (j, relation)
                stack = stack[1:]
            elif t[0] == 'Right-Arc':
                relation = t[1]
                parent[stack[0]] = stack[1]
                output[stack[0]] = "%d\t%s" % (stack[1], relation)
                stack = stack[1:]
            elif t[0] == 'Shift':
                pos[j] = t[1]
                stack = [j] + stack
                buf = buf[1:]

    if "" in output[1:]:
        print "\n".join(["\t".join(x) for x in lines])
        print "\n".join(["%s\t%s" % (x=='', x) for x in output[1:]])
    fout.write("%s\n\n" % ("\n".join("\t".join(t) for t in zip(pos[1:], output[1:]))))

lines = []

fout = open(args.fileout, 'w')
with open(args.filein, 'r') as fin:
    line = fin.readline()
    while line:
        line = line.strip().split()

        if len(line) == 0:
            processlines(lines, fout)
            lines = []
        else:
            lines += [line]

        line = fin.readline()

    if len(lines) > 0:
        processlines(lines, fout)

fout.close()
