#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import click
import os
from ruamel.yaml import YAML
from pathlib import Path
from cnn.model import TextCNN
from cnn.utils import get_now

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # {0, 1, 2, 3}


@click.command()
@click.option('--data-cnf', help='dataset config')
@click.option('--model-cnf', help='model config')
def main(data_cnf, model_cnf):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model_path = os.path.join(model_cnf['path'], model_cnf['name'] + '-' + data_cnf['name'])
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    model = TextCNN(**model_cnf['model'], model_path=model_path)

    model_name = model_cnf['name']
    print('model name:', model_name)

    print(get_now(), 'loading dataset')
    train_x, train_y = np.load(data_cnf['train']['inputs']), np.load(data_cnf['train']['labels'])
    valid_x, valid_y = np.load(data_cnf['valid']['inputs']), np.load(data_cnf['valid']['labels'])

    print('size of train set:', len(train_x))
    print('size of valid set:', len(valid_x))

    model.train(sess, train_x, train_y, valid_x, valid_y, **model_cnf['train'])


if __name__ == '__main__':
    main()
