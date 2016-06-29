#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

import numpy

class ELU(function.Function):
    """Exponential Linear Unit."""

    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
                x_type.dtype == numpy.float32,
                )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
                'T x, T alpha', 'T y',
                'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
                'T x, T gy, T alpha', 'T gx',
                'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)


class Generator(chainer.Chain):
    def __init__(self, nz=30):
        super(Generator, self).__init__(
                l0z=L.Linear(nz, 6 * 6 * 128, wscale=0.02 * math.sqrt(nz)),
                dc1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
                dc2=L.Deconvolution2D(64, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
                dc3=L.Deconvolution2D(32, 1, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
                bn0l=L.BatchNormalization(6 * 6 * 128),
                bn0=L.BatchNormalization(128),
                bn1=L.BatchNormalization(64),
                bn2=L.BatchNormalization(32)
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 128, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        x = (self.dc3(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
                c0=L.Convolution2D(1, 32, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 3)),
                c1=L.Convolution2D(32, 64, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 32)),
                c2=L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02 * math.sqrt(4 * 4 * 64)),
                l4l=L.Linear(6 * 6 * 128, 2, wscale=0.02 * math.sqrt(6 * 6 * 128)),
                bn0=L.BatchNormalization(32),
                bn1=L.BatchNormalization(64),
                bn2=L.BatchNormalization(128)
        )

    def __call__(self, x, test=False):
        h = elu(self.c0(x))  # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        l = self.l4l(h)
        return l


