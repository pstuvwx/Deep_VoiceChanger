import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from .sn_convolution_nd import SNConvolutionND
from .sn_linear import SNLinear

def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))

def _downsample(x):
    return F.average_pooling_2d(x, 2)

def upsample_conv(x, conv):
    return conv(_upsample(x))

def _upsample_frq(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, (1,2), outsize=(h, w * 2))

def _downsample_frq(x):
    return F.average_pooling_2d(x, (1,2))

def upsample_conv_frq(x, conv):
    return conv(_upsample_frq(x))

class ResBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu, mode='none', bn=True, dr=None):
        super(ResBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.mode = _downsample if mode == 'down' else _upsample if mode == 'up' else None
        self.learnable_sc = in_channels != out_channels
        self.dr = dr
        self.bn = bn
        with self.init_scope():
            self.c1 = L.Convolution1D(in_channels,  out_channels, ksize=ksize, pad=pad, initialW=initializer, nobias=bn)
            self.c2 = L.Convolution1D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer, nobias=bn)
            if bn:
                self.b1 = L.BatchNormalization(out_channels)
                self.b2 = L.BatchNormalization(out_channels)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        if self.bn:
            h = self.b1(h)
        if self.activation:
            h = self.activation(h)
        if self.mode:
            h = self.mode(h)
        if self.dr:
            with chainer.using_config('train', True):
                h = F.dropout(h, self.dr)
        h = self.c2(h)
        if self.bn:
            h = self.b2(h)
        if self.activation:
            h = self.activation(h)
        return h

    def shortcut(self, x):
        if self.mode:
            x = self.mode(x)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class ConvBlock(chainer.Chain):
    #O = (I+2P-K)/S+1
    def __init__(self, in_channels, out_channels, mode='none', activation=F.leaky_relu, bn=True, dr=None):
        super(ConvBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.bn = bn
        self.dr = dr
        with self.init_scope():
            if mode == 'none':
                self.c = L.Convolution1D(in_channels, out_channels, ksize=3, stride=1, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'down':
                self.c = L.Convolution1D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'up':
                self.c = L.Deconvolution1D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            else:
                raise Exception('mode is missing')
            if bn:
                self.b = L.BatchNormalization(out_channels)

    def __call__(self, h):
        if self.dr:
            with chainer.using_config('train', True):
                h = F.dropout(h, self.dr)
        h = self.c(h)
        if self.bn:
            h = self.b(h)
        if self.activation:
            h = self.activation(h)
        return h
    
class SNResBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.relu, sample='none'):
        super(SNResBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.sample = _downsample if sample == 'down' else _upsample if sample == 'up' else None
        self.learnable_sc = in_channels != out_channels or sample == 'down' or sample == 'up'
        with self.init_scope():
            self.c1 = SNConvolutionND(1, in_channels,  out_channels, ksize=3, pad=1, initialW=initializer)
            self.c2 = SNConvolutionND(1, out_channels, out_channels, ksize=3, pad=1, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = SNConvolutionND(1, in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.sample:
            h = self.sample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.sample:
                return self.sample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class SNConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu, mode='none'):
        super(SNConvBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            if mode == 'none':
                self.c = SNConvolutionND(1, in_channels,  out_channels, ksize=3, stride=1, pad=1, initialW=initializer)
            elif mode == 'down':
                self.c = SNConvolutionND(1, in_channels,  out_channels, ksize=4, stride=2, pad=1, initialW=initializer)
            elif mode == 'sample-down':
                self.c = SNConvolutionND(1, in_channels,  out_channels, ksize=4, stride=2, pad=1, initialW=initializer)
                self.activation = lambda x: _downsample(activation(x))

    def __call__(self, x):
        h = x
        h = self.c(h)
        h = self.activation(h)
        return h

class SNLinearBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu):
        super(SNLinearBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.l = SNLinear(in_channels,  out_channels, initialW=initializer)

    def __call__(self, x):
        h = x
        h = self.l(h)
        h = self.activation(h)
        return h

class SNMDBlock(chainer.Chain):
    def __init__(self, in_channels, in_size=4, B=100, C=5):
        super(SNMDBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.B = B
        self.C = C
        with self.init_scope():
            self.l = SNLinear(in_size * in_channels + B, 1, initialW=initializer)
            self.md = SNLinear(in_size * in_channels, B * C, initialW=initializer)

    def __call__(self, x):
        N = x.shape[0]
        #Below code copyed from https://github.com/pfnet-research/chainer-gan-lib/blob/master/minibatch_discrimination/net.py
        feature = F.reshape(x, (N, -1))
        m = F.reshape(self.md(feature), (N, self.B * self.C, 1))
        m0 = F.broadcast_to(m, (N, self.B * self.C, N))
        m1 = F.transpose(m0, (2, 1, 0))
        d = F.absolute(F.reshape(m0 - m1, (N, self.B, self.C, N)))
        d = F.sum(F.exp(-F.sum(d, axis=2)), axis=2) - 1
        h = F.concat([feature, d])

        h = self.l(h)
        return h

