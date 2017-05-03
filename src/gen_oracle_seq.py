import sys
from parserstate import ParserState
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid
import argparse
from utils import read_mappings, transsys_lookup, read_gold_parserstates, write_gold_trans

parser = argparse.ArgumentParser()

parser.add_argument('filein', type=str, help="Input CONLL file")
parser.add_argument('fileout', type=str, help="Output oracle sequence file")
parser.add_argument('--transsys', type=str, choices=['ASw', 'AER', 'AES', 'ASd', 'AH'], help="Transition system to use", default="ASw")
parser.add_argument('--mappings', default="mappings.txt", type=str, help="Mapping file for the dataset")
parser.add_argument('--fpos', default=False, action='store_true', help="Include fine-grained POS")

args = parser.parse_args()

mappings, invmappings = read_mappings(args.mappings, transsys_lookup(args.transsys))
transsys = transsys_lookup(args.transsys)(mappings, invmappings)

with open(args.filein, 'r') as fin:
    gold_states = read_gold_parserstates(fin, transsys, args.fpos)

with open(args.fileout, 'w') as fout:
    for state in gold_states:
        write_gold_trans(state, fout)
