#!/bin/bash
if [ -f 'corenlp.env' ]; then
    source corenlp.env
else
    echo "Please run setup_corenlp.sh first!"
    exit
fi

for split in train dev test; do
    echo Converting $split split...
    java -mx1g -cp ${CORENLP_HOME} edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile ptb3-wsj-${split}.trees -checkConnected -basic -keepPunct -conllx > ptb3-wsj-${split}.conllx
done
