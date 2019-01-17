# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from layer.reorg_layer import reorg_layer

_activation_dict = {
    'leaky': tf.nn.leaky_relu,
    'relu': tf.nn.relu
}


# cfg_layerName(B, H, W, C, net, param, weights_walker, stack, scope=None):
#    pass

def cfg_net(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    width = int(param["width"])
    height = int(param["height"])
    channels = int(param["channels"])
    net = tf.placeholder(tf.float32, [None, width, height, channels], name=scope)
    return net


def cfg_convolutional(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    batch_normalize = 'batch_normalize' in param
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])
    activation = None
    weight_size = C * filters * size * size

    if "activation" in param:
        activation = _activation_dict.get(param['activation'], None)

    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(param['name'],
                                  filters=filters,
                                  weight_size=weight_size,
                                  batch_normalize=batch_normalize)
    weights = weights.reshape(C, filters, size, size).transpose([2, 3, 1, 0])

    conv_args = {
        "num_outputs": filters,
        "kernel_size": size,
        "stride": stride,
        "activation_fn": activation
    }

    if const_inits:
        conv_args.update({
            "weights_initializer": tf.initializers.constant(weights),
            "biases_initializer": tf.initializers.constant(biases)
        })
        
    if batch_normalize:
        conv_args.update({
            "normalizer_fn": slim.batch_norm
        })

        if const_inits:
            conv_args.update({
                "normalizer_params": {
                    "param_initializers": {
                        "gamma": tf.initializers.constant(scales),
                        "moving_mean": tf.initializers.constant(rolling_mean),
                        "moving_variance": tf.initializers.constant(rolling_variance)
                    }
                }
        })

    net = slim.conv2d(net, scope=scope, **conv_args)
    return net


def cfg_maxpool(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    pool_args = {
        "kernel_size": int(param['size']),
        "stride": int(param['stride'])
    }

    net = slim.max_pool2d(net, scope=scope, **pool_args)
    return net


def cfg_route(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    if not isinstance(param["layers"], list):
        param["layers"] = [param["layers"]]
    net_index = [int(x) for x in param["layers"]]
    nets = [stack[x] for x in net_index]

    net = tf.concat(nets, axis=-1, name=scope)
    return net


def cfg_reorg(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    reorg_args = {
        "stride": int(param['stride'])
    }

    net = reorg_layer(net, name=scope, **reorg_args)
    return net


def cfg_ignore(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose):
    if verbose:
        print("=> Ignore: ", param)

    return net


_cfg_layer_dict = {
    "net": cfg_net,
    "convolutional": cfg_convolutional,
    "maxpool": cfg_maxpool,
    "route": cfg_route,
    "reorg": cfg_reorg
}


def get_cfg_layer(net, layer_name, param, weights_walker, stack, scope=None, const_inits=True, verbose=True):
    B, H, W, C = [None, None, None, None] if net is None else net.shape.as_list()
    layer = _cfg_layer_dict.get(layer_name, cfg_ignore)(B, H, W, C, net, param, weights_walker, stack, scope, const_inits, verbose)
    return layer
