# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from util.cfg_layer import get_cfg_layer
from util.reader import WeightsReader, CFGReader


def parse_net(num_layers, cfg, weights):
    net = None
    counters = {}
    stack = []
    cfg_walker = CFGReader(cfg)
    weights_walker = WeightsReader(weights)

    for ith, layer in enumerate(cfg_walker):
        if ith > num_layers and num_layers > 0:
            break

        layer_name = layer['name']
        counters.setdefault(layer_name, 0)
        counters[layer_name] += 1

        scope = "{}{}{}".format(args.prefix, layer['name'], counters[layer_name])

        net = get_cfg_layer(net, layer_name, layer, weights_walker, stack, scope=scope)

        stack.append(net)
        print(ith, net)


def main(args):
    output_file = os.path.join(args.output, os.path.splitext(os.path.split(args.cfg)[-1])[0] + ".ckpt")
    parse_net(args.layers, args.cfg, args.weights)
    saver = tf.train.Saver(slim.get_model_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, output_file)


if "__main__" == __name__:
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--cfg', default="data/yolo.cfg", help='CFG File')
    parser.add_argument('--weights', default='data/yolo.weights', help='Weight file')
    parser.add_argument('--output', default='data/', help='Output folder')
    parser.add_argument('--prefix', default='yolo/', help='Layer name prefix')
    parser.add_argument('--layers', default=0, help='How many layers, 0 means all')
    parser.add_argument('--gpu', '-g', default='0', help='GPU')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)

    main(args)
