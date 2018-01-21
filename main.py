# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from layer.reorg_layer import reorg_layer

from reader.cfg_reader import CFGReader
from reader.weights_reader import WeightsReader


def parse(cfg, weights):
    net = None
    counters = {}
    stack = []
    cfg_walker = CFGReader(cfg)
    weights_walker = WeightsReader(weights)

    for ith, layer in enumerate(cfg_walker):
        if net is None:
            B, H, W, C = [None, None, None, None]
        else:
            B, H, W, C = net.shape.as_list()

        if layer['name'] not in counters:
            counters[layer['name']] = 0
        counters[layer['name']] += 1

        if "[net]" == layer['name']:
            width = int(layer["width"])
            height = int(layer["height"])
            channels = int(layer["channels"])
            net = tf.placeholder(tf.float32, [None, width, height, channels])

        elif "[convolutional]" == layer['name']:
            batch_normalize = 'batch_normalize' in layer
            size = int(layer['size'])
            filters = int(layer['filters'])
            stride = int(layer['stride'])
            activation = None
            if "activation" in layer:
                if 'leaky' == layer['activation']:
                    activation = tf.nn.leaky_relu
                elif 'relu' == layer['activation']:
                    activation = tf.nn.relu

            weight_size = C * filters * size * size
            biases, scales, rolling_mean, rolling_variance, weights = \
                weights_walker.get_weight(layer['name'],
                                          filters=filters,
                                          weight_size=weight_size,
                                          batch_normalize=batch_normalize)
            weights = weights.reshape(
                C, filters, size, size).transpose([2, 3, 1, 0])

            conv_args = dict(
                num_outputs=filters,
                kernel_size=size,
                stride=stride,
                activation_fn=activation,
                weights_initializer=tf.initializers.constant(weights),
                biases_initializer=tf.initializers.constant(biases),
                scope="{}{}{}".format(args.prefix, "/convolutional", counters[layer['name']])
            )
            if batch_normalize:
                conv_args.update({
                    "normalizer_fn": slim.batch_norm,
                    "normalizer_params": {"param_initializers": {
                        "gamma": tf.initializers.constant(scales),
                        "moving_mean": tf.initializers.constant(rolling_mean),
                        "moving_variance": tf.initializers.constant(rolling_variance),
                    }},
                })

            net = slim.conv2d(net, **conv_args)

        elif "[maxpool]" == layer['name']:
            size = int(layer['size'])
            stride = int(layer['stride'])
            net = slim.max_pool2d(net,
                                  kernel_size=size,
                                  stride=stride,
                                  scope="{}{}{}".format(args.prefix, "/maxpool", counters[layer['name']]))

        elif "[route]" == layer['name']:
            if not isinstance(layer["layers"], list):
                layer["layers"] = [layer["layers"]]
            net_index = [int(x) for x in layer["layers"]]
            nets = [stack[x] for x in net_index]
            net = tf.concat(nets, axis=-1)

        elif "[reorg]" == layer['name']:
            stride = int(layer['stride'])
            net = reorg_layer(net, stride)

        else:
            print("Ignore: ", layer)

        stack.append(net)
        print(ith, net)


def main(args):
    output_file = os.path.join(args.output, os.path.splitext(
        os.path.split(args.cfg)[-1])[0] + ".ckpt")
    parse(args.cfg, args.weights)
    saver = tf.train.Saver(slim.get_model_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, output_file)


if "__main__" == __name__:
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--cfg', default="data/yolo.cfg", help='CFG File')
    parser.add_argument('--weights', default='data/yolo.weights', help='Weight file')
    parser.add_argument('--output', default='data', help='Output folder')
    parser.add_argument('--prefix', default='yolo', help='Layer prefix')
    parser.add_argument('--gpu', '-g', default='0', help='GPU')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)

    main(args)
