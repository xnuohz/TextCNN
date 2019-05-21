#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import click
import numpy as np
import os
from tensorflow.contrib import learn
from ruamel.yaml import YAML
from pathlib import Path
from sklearn.model_selection import train_test_split
from cnn.utils import load_data_and_labels


@click.command()
@click.option('--data-cnf', help='dataset config')
def main(data_cnf):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))

    print('load data')
    x_text, y = load_data_and_labels(*data_cnf['source'])

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    data_x = x[shuffle_indices]
    data_y = y[shuffle_indices]

    # Split train/test set
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=data_cnf['valid']['size'])

    np.save(data_cnf['train']['inputs'], train_x)
    np.save(data_cnf['train']['labels'], train_y)
    np.save(data_cnf['valid']['inputs'], valid_x)
    np.save(data_cnf['valid']['labels'], valid_y)
    vocab_processor.save(os.path.join(data_cnf['path'], "vocab"))

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(train_y), len(valid_y)))


if __name__ == '__main__':
    main()
