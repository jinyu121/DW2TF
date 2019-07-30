# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import os
import tensorflow as tf

from util.cfg_layer import get_cfg_layer
from util.reader import WeightsReader, CFGReader


def parse_net(num_layers, cfg, weights, training=False, const_inits=True, verbose=True):
    net = None
    counters = {}
    stack = []
    cfg_walker = CFGReader(cfg)
    weights_walker = WeightsReader(weights)
    output_index = []
    num_layers = int(num_layers)

    for ith, layer in enumerate(cfg_walker):
        if ith > num_layers and num_layers > 0:
            break
        layer_name = layer['name']
        counters.setdefault(layer_name, 0)
        counters[layer_name] += 1
        scope = "{}{}{}".format(args.prefix, layer['name'], counters[layer_name])
        net = get_cfg_layer(net, layer_name, layer, weights_walker, stack, output_index, scope,
                            training=training, const_inits=const_inits, verbose=verbose)
        # Exclude `net` layer from stack (for correct layer indexing)
        # See https://github.com/jinyu121/DW2TF/issues/30
        # See https://github.com/AlexeyAB/darknet/issues/487#issuecomment-374902735
        if layer['name'] != 'net':
            stack.append(net)
        if verbose:
            print(ith, net)

    if verbose:        
        for ind in output_index:
            print("=> Output layer: ", stack[ind])


def main(args):
    ckpt_path = os.path.join(args.output, os.path.splitext(os.path.split(args.cfg)[-1])[0] + ".ckpt")
    pb_path = os.path.join(args.output, os.path.splitext(os.path.split(args.cfg)[-1])[0] + ".pb")

    # ----------------------------------------------------------
    # Save temporary .ckpt from graph containing pre-trained
    # weights as const initializers. This is not portable as
    # graph.pb or graph.meta is huge (contains weights).
    # ----------------------------------------------------------
    tf.reset_default_graph()
    parse_net(args.layers, args.cfg, args.weights, args.training)
    graph = tf.get_default_graph()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckpt_path, write_meta_graph=False)

    # ----------------------------------------------------------
    # Save .pb, .meta and final .ckpt by restoring weights
    # from previous .ckpt into the new (compact) graph.
    # ----------------------------------------------------------
    tf.reset_default_graph()
    parse_net(args.layers, args.cfg, args.weights, args.training, const_inits=False, verbose=False)
    graph = tf.get_default_graph()

    with tf.gfile.GFile(pb_path, 'wb') as f:
        f.write(graph.as_graph_def(add_shapes=True).SerializeToString())
    print("Saved .pb to '{}'".format(pb_path))

    with tf.Session(graph=graph) as sess:
        # Load weights (variables) from earlier .ckpt before saving out
        var_list = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        for key in reader.get_variable_to_shape_map():
            # Look for all variables in ckpt that are used by the graph
            try:
                tensor = graph.get_tensor_by_name(key + ":0")
            except KeyError:
                # This tensor doesn't exist in the graph (for example it's
                # 'global_step' or a similar housekeeping element) so skip it.
                continue
            var_list[key] = tensor
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt_path)

        saver.export_meta_graph(ckpt_path+'.meta', clear_devices=True, clear_extraneous_savers=True)
        print("Saved .meta to '{}'".format(ckpt_path+'.meta'))

        saver.save(sess, ckpt_path, write_meta_graph=False)
        print("Saved .ckpt to '{}'".format(ckpt_path))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cfg', default='data/network.cfg', help='Darknet .cfg file')
    parser.add_argument('--weights', default='data/network.weights', help='Darknet .weights file')
    parser.add_argument('--output', default='data/', help='Output folder')
    parser.add_argument('--prefix', default='network/', help='Import scope prefix')
    parser.add_argument('--layers', default=0, help='How many layers, 0 means all')
    parser.add_argument('--gpu', '-g', default='', help='GPU')
    parser.add_argument('--training', dest='training', action='store_true', help='Save training mode graph')
    args = parser.parse_args()

    # Set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    # Filter out TensorFlow INFO and WARNING logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

    main(args)
