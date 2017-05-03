#!/bin/bash
version=$1

if [ -z $version ]; then
    echo "Usage: ./setup_corenlp.sh <version>"
    echo "Only version 3.3 is supported right now"
    exit
fi

function download_and_setup() {
    prefix=$1
    if [ ! -f "${prefix}.zip" ]; then
        wget http://nlp.stanford.edu/software/${prefix}.zip
    fi
    if [ ! -d $prefix ]; then
        unzip ${prefix}.zip
    fi
    echo "export CORENLP_HOME=`pwd`/${prefix}/stanford-corenlp-?.?.?.jar" > corenlp.env
}

if [ $version == '3.3' ]; then
    download_and_setup 'stanford-corenlp-full-2013-11-12'
fi
