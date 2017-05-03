#!/bin/bash

conllfile=$1

echo '::rel'
cat $conllfile | grep -v '^#' | grep -v '^[0-9]*\-' | grep '\S' | cut -d'	' -f8 | sort | uniq

echo '::pos'
cat $conllfile | grep -v '^#' | grep -v '^[0-9]*\-' | grep '\S' | cut -d'	' -f4 | sort | uniq

echo '::fpos'
cat $conllfile | grep -v '^#' | grep -v '^[0-9]*\-' | grep '\S' | cut -d'	' -f5 | sort | uniq
