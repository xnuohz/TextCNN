# TextCNN

Tensorflow implementation of "Convolutional Neural Networks for Sentence Classification"

![model](asserts/model.png)

## Requirements

* python 3.6
* tensorflow 1.12.0
* tqdm 4.28.1
* click 7.0
* YAML

## Usage

* preprocess
```bash
python -m cnn.preprocess --data-cnf ../config/dataset/rt-polarity.yaml
```
* training
```bash
python -m cnn.main --data-cnf ../config/dataset/rt-polarity.yaml --model-cnf ../config/model/CNN.yaml
```

## Result

![terminal](asserts/result.png)

## Reference

* [Paper - Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
* [Tutorial - Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)