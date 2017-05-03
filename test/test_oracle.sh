#!/bin/bash

# This script verifies the implementation of a transition system by running
# them on a dataset, and verifying the oracle sequence with a more transparent
# implementation of it. The first argument is the dataset (.conll* file) to
# run the test on, the second argument is the acronym for the transition
# system to be tested, and the third argument is the mappings files. For the
# latter two please refer to the README.md file in the root directory of this
# repository. For the first argument, note that the dataset should contain only
# projective parses for all tests to pass.
#
# This script should be run from the root directory of the repository.

input=$1
transsys=$2
mappings=$3

cd src
python gen_oracle_seq.py ../$input ../${input}.seq --transsys $transsys --mappings ../$mappings
cd ..
python test/test_oracle.py ${input}.seq ${input}.parse --transsys $transsys

echo "Taking the diff, nothing should be output"
# diff
diff <(cut -d'	' -f4 $input) <(cut -d'	' -f1 ${input}.parse)
diff <(cut -d'	' -f7-8 $input) <(cut -d'	' -f2-3 ${input}.parse)

# clean up
rm ${input}.seq
rm ${input}.parse

echo "Done!"
