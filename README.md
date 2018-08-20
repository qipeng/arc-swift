## Arc-swift
This repository contains an implementation of the parsers described in [Arc-swift: A Novel Transition System for Dependency Parsing](https://nlp.stanford.edu/pubs/qi2017arcswift.pdf). If you use arc-swift in your work, please cite us with the BibTeX item below.

```
@inproceedings{qi2017arcswift,
  title={Arc-swift: A Novel Transition System for Dependency Parsing},
  author={Qi, Peng and Manning, Christopher D.},
  booktitle={Proceedings of the 55th Annual Meeting of Association for Computational Linguistics},
  year={2017}
}
```

Running the parsers requires Tensorflow 1.0 or above. The data preparation script also requires a working Java 8 installation to run [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html). Other Python dependencies are included in `requirements.txt`, and can be installed via `pip` by running

```
pip install -r requirements.txt
```

## Train your own parsers
### Data preparation

##### Penn Treebank

We provide code for converting the Wall Street Journal section of Penn Treebank into [Stanford Dependencies](https://nlp.stanford.edu/software/stanford-dependencies.shtml).

To use the code, you first need to obtain the corresponding parse trees from LDC, make the standard train/dev/test split (Sections 02-21 for training, 22 for development, and 23 for testing). Copy the splits to `utils/ptb3-wsj` and name them `ptb3-wsj-train.trees`, `ptb3-wsj-dev.trees`, and `ptb3-wsj-test.trees`, respectively, then run the following scripts

```
./setup_corenlp.sh 3.3
./convert_splits_to_depparse.sh
```

The first script downloads [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) v3.3.0 in this directory, which is necessary for converting Penn Treebank parse trees to dependency parses in the second script.

To make the parse trees available for training, it is necessary to keep projective trees only in the training set. For this, go to `utils/` and run

```
python filter_nonproj.py ptb3-wsj/ptb3-wsj-train.conllx ../src/ptb3-wsj-train.conllx
```

The dev and test sets shouldn't be altered, so we just copy them directly to the `src` directory for later use.

```
cp ptb3-wsj/ptb3-wsj-dev.conllx ../src
cp ptb3-wsj/ptb3-wsj-test.conllx ../src
```

As a final step in data preparation, we would need to create a file that maps all dependency arc types and all part-of-speech (POS) types into integers. This can be achieved by running the following script under `utils/`

```
./create_mappings.sh ptb3-wsj/ptb3-wsj-train.conllx > ../src/mappings-ptb.txt
```

To train the parser, we would also need to create the oracle sequence of transitions for our parsers to follow. To do this, go to `src/`, and run

```
python gen_oracle_seq.py ptb3-wsj-train.conllx train.ASw.seq --transsys ASw --mappings mappings-ptb.txt
```

Here, `mappings-ptb.txt` is the mappings file we just created, `ASw` stands for _arc-swift_, and `train.ASw.seq` is the output file containing oracle transitions for the training data.

##### Universal Dependencies

The processing steps of Universal Dependencies trees and that of Penn Treebank parses is very similar, modulo that conversion to dependency parses is not necessary. The Universal Dependencies v1.3 data used in the paper can be found [here](http://universaldependencies.org/#download).

### Training

To train the parsers, you might want to download the pretrained [GloVe vectors](https://nlp.stanford.edu/projects/glove/). For the experiments in the paper, we used the 100-dimensional embeddings trained on Wikipedia and Gigaword (glove.6B.zip). Download and unzip the GloVe files in `src`, and to train the _arc-swift_ parser, simply run

```
mkdir ASw_models
python train_parser.py ptb3-wsj-train.conllx --seq_file train.ASw.seq --wordvec_file glove.6B.100d.txt --vocab_file vocab.pickle --feat_file ptb_ASw_feats.pickle --model_dir ASw_models --transsys ASw --mappings_file mappings-ptb.txt
```

Note that if you're using the GPU, you might want to specify `CUDA_VISIBLE_DEVICES` to tell Tensorflow which GPU device to use. `vocab.pickle` and `ptb_ASw_feats.pickle` are files that the training code will automatically generate and reuse should you want to train the parser more than one time with the same data. For more arguments the training code supports, run

```
python train_parser.py -h
```

To train parsers for other transition systems, simply replace the `--transsys` argument with the short name for the transition system you are interested in.

|Short name |Transition system |
|-----------|------------------|
|ASw        | arc-swift        |
|ASd        | arc-standard     |
|AES        | arc-eager-Shift  |
|AER        | arc-eager-Reduce |
|AH         | arc-hybrid       |

To train the parsers on Universal Dependencies English Treebank, `--epoch_multiplier 3` should also be used to reproduce the training settings described in the paper.

### Evaluation

To evaluate the trained parsers, run

```
python eval_parser.py ptb3-wsj-dev.conllx --vocab_file vocab.pickle --model_dir ASw_models --transsys ASw --eval_dataset dev --mappings_file mappings-ptb
```

This will generate output files in the model directory with names like `models_ASw/dev_eval_beam_1_output_epoch0.txt`, which contain the predicted dependency parses.

We also provide a python implementation of labelled and unlabelled attachment score evaluation. The interface is very similar to the CoNLL official script, simply run

```
cut -d"	" -f1-6 ptb3-wsj-dev.conllx| paste - ASw_models/dev_eval_beam_1_output_epoch0.txt > system_pred.conllx
python eval.py -g ptb3-wsj-dev.conllx -s system_pred.conllx
```
where `-g` stands for gold file, and `-s` stands for system prediction. Note that the delimiter in the call to `cut` is a tab character. By default the script removes punctuation according to the gold Penn Treebank POS tags of the tokens. To run generate results compatible with the CoNLL official evaluation script, make sure to use `--language conll`.

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
