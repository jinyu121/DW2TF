# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class WeightsReader:
    def __init__(self, path):
        self.eof = False  # end of file
        self.path = path  # current pos
        if path is None:
            self.eof = True
            return
        else:
            self.size = os.path.getsize(path)  # save the path
            major, minor, revision, seen = np.memmap(path, shape=(), mode='r', offset=0, dtype='({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def _read(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, 'Over-read {}'.format(self.path)

        float32_1d_array = np.memmap(
            self.path, shape=(), mode='r',
            offset=self.offset,
            dtype='({})float32,'.format(size)
        )

        self.offset = end_point
        if end_point == self.size:
            self.eof = True
        return float32_1d_array

    def get_weight_convolutional(self, filters, weight_size, batch_normalize=False):
        biases = self._read(filters)
        scales = self._read(filters) if batch_normalize else None
        rolling_mean = self._read(filters) if batch_normalize else None
        rolling_variance = self._read(filters) if batch_normalize else None
        weights = self._read(weight_size)
        return biases, scales, rolling_mean, rolling_variance, weights

    def get_weight(self, name, **args):
        if "[convolutional]" == name:
            return self.get_weight_convolutional(**args)
        return None