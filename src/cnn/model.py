#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tqdm import trange
from cnn.utils import get_now


class TextCNN(object):
    """
    A CNN for text classification
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 learning_rate, model_path):
        """
        :param sequence_length: The length of sentences.
        :param num_classes: Number of classes in the output layer.
        :param vocab_size: The size of vocabulary. [vocab_size, embedding_size]
        :param embedding_size: The dimensionality of embeddings.
        :param filter_sizes: The number of words we want our convolutional filters to cover.
        :param num_filters: The number of filters per filter size.
        :param learning_rate: Optimizer params.
        :param model_path: The path that model params will be saved.
        """
        self.model_path = model_path
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), name='w')
            # [None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(w, self.input_x)
            # [None, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                tf.summary.histogram('conv-maxpool-%s/Weights' % filter_size, w)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                tf.summary.histogram('conv-maxpool-%s/bias' % filter_size, b)
                conv = tf.nn.conv2d(self.embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        num_filter_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

        # Add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope('output'):
            self.scores = tf.layers.dense(self.h_drop, num_classes, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', self.loss)

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

    def train_step(self, sess, train_x, train_y):
        feed_dict = {
            self.input_x: train_x,
            self.input_y: train_y,
            self.dropout_keep_prob: 0.5
        }
        return sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

    def valid_step(self, sess, valid_x, valid_y):
        feed_dict = {
            self.input_x: valid_x,
            self.input_y: valid_y,
            self.dropout_keep_prob: 1
        }
        return sess.run([self.accuracy, self.loss], feed_dict=feed_dict)

    def train(self, sess, train_x, train_y, valid_x, valid_y, epoch=20, batch_size=128):
        # writer = tf.summary.FileWriter(self.model_path, sess.graph)
        print(get_now(), 'start training')
        train_idx = sorted(range(len(train_x)), key=lambda x: len(train_x[x]), reverse=True)
        valid_idx = sorted(range(len(valid_x)), key=lambda x: len(valid_x[x]), reverse=True)
        sess.run(tf.global_variables_initializer())
        best_accuracy = 0
        epochs = trange(epoch, desc='Accuracy and Loss')
        for _ in epochs:
            train_loss = 0
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i:i + batch_size]
                _, loss = self.train_step(sess, train_x[batch_idx], train_y[batch_idx])
                train_loss += loss * batch_size
            train_loss /= len(train_idx)

            valid_loss = valid_accuracy = 0
            for j in range(0, len(valid_idx), batch_size):
                valid_batch_idx = valid_idx[j:j + batch_size]
                accuracy, loss = self.valid_step(sess, valid_x[valid_batch_idx], valid_y[valid_batch_idx])
                valid_accuracy += accuracy * batch_size
                valid_loss += loss * batch_size
            valid_loss /= len(valid_idx)
            valid_accuracy /= len(valid_idx)
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                self.saver.save(sess, self.model_path)

            desc = 'train loss: {}, valid loss: {}, Accuracy: {}'.format(round(train_loss, 5), round(valid_loss, 5),
                                                                         round(valid_accuracy, 5))
            epochs.set_description(desc)
        print('best accuracy: ', best_accuracy)
