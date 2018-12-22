import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from .sn_convolution_2d import SNConvolution2D, SNDeconvolution2D
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
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.leaky_relu, mode='none', bn=False, dr=None):
        super(ResBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.mode = _downsample if mode == 'down' else _upsample if mode == 'up' else None
        self.learnable_sc = in_channels != out_channels
        self.dr = dr
        self.bn = bn
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels,  out_channels, ksize=ksize, pad=pad, initialW=initializer, nobias=bn)
            self.c2 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer, nobias=bn)
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
    def __init__(self, in_channels, out_channels, mode='none', activation=F.leaky_relu, bn=False, dr=None):
        super(ConvBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.bn = bn
        self.dr = dr
        with self.init_scope():
            if mode == 'none':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'none-7':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=(7,7), stride=1, pad=(3,3), initialW=initializer, nobias=bn)
            elif mode == 'down':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'up':
                self.c = L.Deconvolution2D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'full-down':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=4, stride=1, pad=0, initialW=initializer, nobias=bn)
            elif mode == 'frq':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
            elif mode == 'frq-down':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
                self.activation = lambda x: activation(_downsample(x))
            elif mode == 'frq-up':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
                self.activation = lambda x: activation(_upsample(x))
            elif mode == 'pad':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=2, initialW=initializer, nobias=bn)
            elif mode == 'trim':
                self.c = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=0, initialW=initializer, nobias=bn)
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
    
class CoPSBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu, bn=True):
        super(CoPSBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.bn = bn
        with self.init_scope():
            self.ps = L.Convolution2D(in_channels, in_channels*4, ksize=1, stride=1, initialW=initializer)
            self.c  = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, initialW=initializer)
            if bn:
                self.b = L.BatchNormalization(out_channels)

    def pixel_shuffle(self, x):
        out = self.ps(x)
        b = out.shape[0]
        c = out.shape[1]
        h = out.shape[2]
        w = out.shape[3]
        out = F.reshape(out, (b, 2, 2, c//4, h, w))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (b, c//4, h*2, w*2))
        return out

    def __call__(self, h):
        h = self.pixel_shuffle(h)
        h = self.c(h)
        if self.bn:
            h = self.b(h)
        if self.activation:
            h = self.activation(h)
        return h
    
class SNResBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu, sample='none', dr=None):
        super(SNResBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.dr = dr
        self.sample = _downsample if sample == 'down' else _upsample if sample == 'up' else None
        self.learnable_sc = in_channels != out_channels or sample == 'down' or sample == 'up'
        with self.init_scope():
            self.c1 = SNConvolution2D(in_channels,  out_channels, ksize=3, pad=1, initialW=initializer)
            self.c2 = SNConvolution2D(out_channels, out_channels, ksize=3, pad=1, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        if self.sample:
            h = self.sample(h)
        if self.dr:
            with chainer.using_config('train', True):
                h = F.dropout(h, self.dr)
        h = self.activation(h)
        h = self.c2(h)
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
    def __init__(self, in_channels, out_channels, mode='none', activation=F.leaky_relu, bn=False, dr=None):
        super(SNConvBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.bn = bn
        self.dr = dr
        with self.init_scope():
            if mode == 'none':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'none-7':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=(7,7), stride=1, pad=(3,3), initialW=initializer, nobias=bn)
            elif mode == 'down':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'up':
                self.c = SNDeconvolution2D(in_channels, out_channels, ksize=4, stride=2, pad=1, initialW=initializer, nobias=bn)
            elif mode == 'full-down':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=4, stride=1, pad=0, initialW=initializer, nobias=bn)
            elif mode == 'frq':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
            elif mode == 'frq-down':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
                self.activation = lambda x: activation(_downsample(x))
            elif mode == 'frq-up':
                self.c = SNConvolution2D(in_channels, out_channels, ksize=(1,9), stride=1, pad=(0,4), initialW=initializer, nobias=bn)
                self.activation = lambda x: activation(_upsample(x))
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
    
class SNLinearBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu, dr=None):
        super(SNLinearBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.dr = dr
        if type(out_channels) is tuple:
            self.out_shape = (-1,)+out_channels
        else:
            self.out_shape = None
        with self.init_scope():
            self.l = SNLinear(in_channels,  np.prod(out_channels), initialW=initializer)

    def __call__(self, x):
        if self.dr:
            x = F.dropout(x, self.dr)
        x = self.l(x)
        x = self.activation(x)
        if self.out_shape:
            x = F.reshape(x, self.out_shape)
        return x

class SNMDBlock(chainer.Chain):
    def __init__(self, in_channels, in_size=4, B=100, C=5, gap=True, dr=None):
        super(SNMDBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.B = B
        self.C = C
        self.dr = dr
        self.gap = gap
        if gap:
            in_size = 1
        if type(in_size) is int:
            in_size = (in_size, in_size)
        with self.init_scope():
            self.l = SNLinear(in_size[0] * in_size[1] * in_channels + B, 1, initialW=initializer)
            self.md = SNLinear(in_size[0] * in_size[1] * in_channels, B * C, initialW=initializer)

    def __call__(self, x):
        if self.dr:
            with chainer.using_config('train', True):
                x = F.dropout(x, self.dr)
        if self.gap:
            x = F.sum(x, axis=(2,3))
        N = x.shape[0]
        #Below code copyed from https://github.com/pfnet-research/chainer-gan-lib/blob/master/minibatch_discrimination/net.py
        feature = F.reshape(F.leaky_relu(x), (N, -1))
        m = F.reshape(self.md(feature), (N, self.B * self.C, 1))
        m0 = F.broadcast_to(m, (N, self.B * self.C, N))
        m1 = F.transpose(m0, (2, 1, 0))
        d = F.absolute(F.reshape(m0 - m1, (N, self.B, self.C, N)))
        d = F.sum(F.exp(-F.sum(d, axis=2)), axis=2) - 1
        h = F.concat([feature, d])

        h = self.l(h)
        return h

class SNL1DBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, width, activation=F.leaky_relu, dr=None):
        super(SNL1DBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.dr = dr
        self.out_ch = out_ch
        with self.init_scope():
            self.l = SNLinear(in_ch*width, out_ch*width, initialW=initializer)

    def __call__(self, x):
        if self.dr:
            x = F.dropout(x, self.dr)
        x = F.transpose(x, (0, 2, 1, 3))
        out_shape = list(x.shape)
        x = F.reshape(x, (-1, x.shape[2]*x.shape[3]))
        x = self.l(x)
        x = self.activation(x)
        out_shape[2] = self.out_ch
        x = F.reshape(x, out_shape)
        x = F.transpose(x, (0, 2, 1, 3))
        return x

class L1DBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, width, activation=F.leaky_relu, dr=None):
        super(L1DBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.dr = dr
        self.out_ch = out_ch
        with self.init_scope():
            self.l = L.Linear(in_ch*width, out_ch*width, initialW=initializer)

    def __call__(self, x):
        if self.dr:
            x = F.dropout(x, self.dr)
        x = F.transpose(x, (0, 2, 1, 3))
        out_shape = list(x.shape)
        x = F.reshape(x, (-1, x.shape[2]*x.shape[3]))
        x = self.l(x)
        x = self.activation(x)
        out_shape[2] = self.out_ch
        x = F.reshape(x, out_shape)
        x = F.transpose(x, (0, 2, 1, 3))
        return x

class CLBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, width, activation=F.leaky_relu, liner_out_ch=1, dr=None):
        super(CLBlock, self).__init__()
        self.dr = dr
        if out_ch - liner_out_ch <= 0:
            raise Exception('out_ch <= liner_out_ch!')
        with self.init_scope():
            self.c = ConvBlock(in_ch, out_ch-liner_out_ch, activation=activation)
            self.l = L1DBlock(in_ch, liner_out_ch, width, activation)

    def __call__(self, x):
        h = x
        if self.dr:
            h = F.dropout(h, self.dr)
        h1 = self.c(h)
        h2 = self.l(h)
        h = F.concat([h1,h2])
        return h

class SNCLBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, width, activation=F.leaky_relu, dr=None):
        super(SNCLBlock, self).__init__()
        self.dr = dr
        with self.init_scope():
            self.c = SNConvBlock(in_ch, out_ch-1, activation=activation)
            self.l = SNL1DBlock(in_ch, 1, width, activation)

    def __call__(self, x):
        h = x
        if self.dr:
            h = F.dropout(h, self.dr)
        h1 = self.c(h)
        h2 = self.l(h)
        h = F.concat([h1,h2])
        return h

