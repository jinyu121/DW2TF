# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class CFGReader:
    def __init__(self, fnm):
        self.fnm = fnm

    def _get_line(self):
        for line in open(self.fnm):
            line = line.split("#")[0].strip()
            if len(line) > 0:
                yield (line.strip())
        yield "[]"

    def get_block(self):
        line_getter = self._get_line()
        obj = None
        while True:
            line = next(line_getter)
            if line.startswith('['):
                if obj:
                    yield (obj)
                    obj = None
                obj = dict()
                obj["name"] = line
            else:
                line = [x.strip() for x in line.split("=")]
                key = line[0]
                value = line[1]
                if ',' in value:
                    value = [x.strip() for x in value.split(",")]
                obj[key] = value

    def __call__(self, *args, **kwargs):
        return self.get_block()

    def __next__(self):
        return self.get_block()

    def __iter__(self):
        return self.get_block()