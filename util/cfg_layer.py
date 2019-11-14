# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layer.reorg_layer import reorg_layer

# From Darknet
_LEAKY_RELU_ALPHA = 0.1
_BATCH_NORM_MOMENTUM = 0.9
_BATCH_NORM_EPSILON = 1e-05


_activation_dict = {
    'leaky': lambda x, y: tf.nn.leaky_relu(x, name=y, alpha=_LEAKY_RELU_ALPHA),
    'relu': lambda x, y: tf.nn.relu(x, name=y),
    'swish': lambda x, y: x * tf.nn.sigmoid(x, name=y),
    'logistic': lambda x, y: tf.nn.sigmoid(x, name=y)
}


# # Syntax
# def cfg_layerName(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
#     pass


def cfg_net(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    width = int(param["width"])
    height = int(param["height"])
    channels = int(param["channels"])
    net = tf.placeholder(tf.float32, [None, width, height, channels], name=scope)
    return net


def cfg_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    batch_normalize = 'batch_normalize' in param
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])

    if 'groups' in param:
        groups = int(param['groups'])
    else:
        groups = 1

    if 'pad' in param:
        pad = 'same' if param['pad'] == '1' else 'valid'
    else:
        pad = 'valid'
    activation = None
    weight_size = C * int(filters / groups) * size * size
    
    if "activation" in param:
        activation = _activation_dict.get(param['activation'], None)

    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(param['name'],
                                  filters=filters,
                                  weight_size=weight_size,
                                  batch_normalize=batch_normalize)
    weights = weights.reshape(int(filters/ groups), C, size, size).transpose([2, 3, 1, 0])

    if groups > 1:
        if 'pad' in param:
            pad = 'SAME' if param['pad'] == '1' else 'VALID'
        else:
            pad = 'VALID'
        conv_args = {
            "strides": [1, stride, stride, 1],
            "padding": pad,
        }

        if stride == 2:
            conv_args.update({
                "padding": 'VALID'
            })
            paddings = tf.constant([[0, 0], [int(size/2), int(size/2)], [int(size/2), int(size/2)], [0, 0]])
            net = tf.pad(net, paddings)

        net = tf.nn.depthwise_conv2d(net, weights, **conv_args)
        if not batch_normalize:
            net = tf.nn.bias_add(net, biases)
    else:
        
        conv_args = {
            "filters": filters,
            "kernel_size": size,
            "strides": stride,
            "activation": None,
            "padding": pad,
        }

        if const_inits:
            conv_args.update({
                "kernel_initializer": tf.initializers.constant(weights, verify_shape=True),
                "bias_initializer": tf.initializers.constant(biases, verify_shape=True)
            })

        if batch_normalize:
            conv_args.update({
                "use_bias": False
            })

        if stride == 2:
            conv_args.update({
                "padding": 'valid'
            })
            paddings = tf.constant([[0, 0], [int(size/2), int(size/2)], [int(size/2), int(size/2)], [0, 0]])
            net = tf.pad(net, paddings)
        
        net = tf.layers.conv2d(net, name=scope, **conv_args)

    if batch_normalize:
        batch_norm_args = {
            "momentum": _BATCH_NORM_MOMENTUM,
            "epsilon": _BATCH_NORM_EPSILON,
            "fused": True,
            "trainable": training,
            "training": training
        }

        if const_inits:
            batch_norm_args.update({
                "beta_initializer": tf.initializers.constant(biases, verify_shape=True),
                "gamma_initializer": tf.initializers.constant(scales, verify_shape=True),
                "moving_mean_initializer": tf.initializers.constant(rolling_mean, verify_shape=True),
                "moving_variance_initializer": tf.initializers.constant(rolling_variance, verify_shape=True)
            })

        net = tf.layers.batch_normalization(net, name=scope+'/BatchNorm', **batch_norm_args)

    if activation:
        net = activation(net, scope+'/Activation')

    return net


def cfg_maxpool(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    pool_args = {
        "pool_size": int(param['size']),
        "strides": int(param['stride']),
        "padding": 'same'
    }

    net = tf.layers.max_pooling2d(net, name=scope, **pool_args)
    return net


def cfg_avgpool(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    # Darknet uses only global avgpool (no stride, kernel size == input size)
    # Reference:
    # https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/avgpool_layer.c#L7
    assert len(param) == 1, "Expected global avgpool; no stride / size param but got param=%s" % param
    pool_args = {
        "pool_size": (H, W),
        "strides": 1
    }

    net = tf.layers.average_pooling2d(net, name=scope, **pool_args)
    return net


def cfg_route(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if not isinstance(param["layers"], list):
        param["layers"] = [param["layers"]]
    net_index = [int(x) for x in param["layers"]]
    nets = [stack[x] for x in net_index]

    net = tf.concat(nets, axis=-1, name=scope)
    return net


def cfg_reorg(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    reorg_args = {
        "stride": int(param['stride'])
    }

    net = reorg_layer(net, name=scope, **reorg_args)
    return net


def cfg_shortcut(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    index = int(param['from'])
    activation = param['activation']
    # assert activation == 'linear'

    from_layer = stack[index]
    net = tf.add(net, from_layer, name=scope)
    return net


def cfg_yolo(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    output_index.append(len(stack) - 1)
    return net


def cfg_upsample(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    stride = int(param['stride'])
    assert stride == 2

    net = tf.image.resize_nearest_neighbor(net, (H * stride, W * stride), name=scope)
    return net


def cfg_softmax(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    net = tf.squeeze(net, axis=[1, 2], name=scope+'/Squeeze')
    net = tf.nn.softmax(net, name=scope+'/Softmax')
    return net


def cfg_ignore(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    if verbose:
        print("=> Ignore: ", param)

    return net
def cfg_scale_channels(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose):
    index = int(param['from'])
    
    from_layer = stack[index]
    net = tf.reshape(net, [-1, 1, 1, net.shape[-1]])
    net = from_layer * net
    return net
    
_cfg_layer_dict = {
    "net": cfg_net,
    "convolutional": cfg_convolutional,
    "maxpool": cfg_maxpool,
    "avgpool": cfg_avgpool,
    "route": cfg_route,
    "reorg": cfg_reorg,
    "shortcut": cfg_shortcut,
    "yolo": cfg_yolo,
    "upsample": cfg_upsample,
    "softmax": cfg_softmax,
    "scale_channels": cfg_scale_channels
}


def get_cfg_layer(net, layer_name, param, weights_walker, stack, output_index,
                  scope=None, training=False, const_inits=True, verbose=True):
    B, H, W, C = [None, None, None, None] if net is None else net.shape.as_list()
    layer = _cfg_layer_dict.get(layer_name, cfg_ignore)(B, H, W, C, net, param, weights_walker, stack, output_index, scope, training, const_inits, verbose)
    return layer