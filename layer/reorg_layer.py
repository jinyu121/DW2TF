# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def none_to_m1(x):
    return x if x is not None else -1


def old_reorg_layer(net, stride=2, name='reorg'):
    batch_size, height, width, channels = net.get_shape().as_list()
    batch_size = none_to_m1(batch_size)
    _height, _width, _channel = height // stride, width // stride, channels * stride * stride
    with tf.name_scope(name):
        net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
        net = tf.transpose(net, [0, 1, 3, 2, 4, 5])  # batch_size, _height, _width, stride, stride, channels
        net = tf.reshape(net, [batch_size, _height, _width, stride * stride * channels], name='reorg')
    return net


def reorg_layer(net, stride=2, name='reorg'):
    with tf.name_scope(name):
        net = tf.extract_image_patches(net,
                                       ksizes=[1, stride, stride, 1],
                                       strides=[1, stride, stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
    return net
