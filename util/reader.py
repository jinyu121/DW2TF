# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ctypes
import numpy as np

import tensorflow as tf


class WeightsReader:
    """
    YOLO .weights file reader
    incremental reader of float32 binary files

    Reference:
    https://github.com/thtrieu/darkflow/blob/b2aee0000cd2a956b9f1de6dbfef94d53158b7d8/darkflow/utils/loader.py#L107-L138
    """

    def __init__(self, path):
        self.eof = False  # end of file
        self.path = path  # current pos
        if path is None:
            self.eof = True
            return
        else:
            self.size = os.path.getsize(path)  # save the path
            major, minor, revision = np.memmap(path, shape=(), mode='r', offset=0, dtype='({})i4,'.format(3))
            self.transpose = major > 1000 or minor > 1000
            if ((major*10 + minor) >= 2 and major < 1000 and minor < 1000):
                self.offset = 12 + ctypes.sizeof(ctypes.c_size_t)
            else:
                self.offset = 16

    def walk(self, size):
        if self.eof:
            return None
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
        biases = self.walk(filters)
        scales = self.walk(filters) if batch_normalize else None
        rolling_mean = self.walk(filters) if batch_normalize else None
        rolling_variance = self.walk(filters) if batch_normalize else None
        weights = self.walk(weight_size)
        return biases, scales, rolling_mean, rolling_variance, weights

    def get_weight(self, name, **args):
        if "convolutional" == name:
            return self.get_weight_convolutional(**args)
        return None


class CFGReader:
    def __init__(self, fnm):
        self.fnm = fnm

    def _get_line(self):
        for line in open(self.fnm):
            line = line.split("#")[0].strip()
            if len(line) > 0:
                yield (line.strip())
        yield "[]"  # Yield a dummy block

    def get_block(self):
        line_getter = self._get_line()
        obj = None
        while True:
            line = next(line_getter)
            if line.startswith('['):
                line = line.strip('[').strip("]")
                if obj:  # Yield previous block first
                    yield (obj)
                # Create a new dict
                obj = dict()
                obj["name"] = line
            else:
                line = [x.strip() for x in line.split("=")]
                key, value = line[0:2]
                if ',' in value:
                    value = [x.strip() for x in value.split(",")]
                obj[key] = value

    def __call__(self, *args, **kwargs):
        return self.get_block()

    def __next__(self):
        return self.get_block()

    def __iter__(self):
        return self.get_block()
