import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

import waver

def add_noise_c(h, sigma=0.2):
    xp = cp.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.data.shape, dtype=cp.float32)
    else:
        return h

def add_noise_n(h, sigma=0.2):
    if chainer.config.train:
        return h + sigma * cp.random.randn(*h.shape, dtype=cp.float32)
    else:
        return h
class CBR(chainer.Chain):
    def __init__(self, in_ch, out_ch, bn=True, activation=F.leaky_relu, mode='down', noise=False):
        super(CBR, self).__init__()
        with self.init_scope():
            self.bn = bn
            self.activation = activation
            self.noise = noise
            w = chainer.initializers.Normal(0.001)
            if mode == 'down':
                self.c = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
            elif mode == 'up':
                self.c = L.Deconvolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
            elif mode == 'none-c':
                self.c = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            elif mode == 'none-d':
                self.c = L.Deconvolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            elif mode == 'full-down':
                self.c = L.Convolution2D(in_ch, out_ch, 4, initialW=w)
            elif mode == 'full-up':
                self.c = L.Deconvolution2D(in_ch, out_ch, 4, initialW=w)
            else:
                raise Exception("mode is missing")
            if bn:
                self.b = L.BatchNormalization(out_ch)
    
    def __call__(self, x, x2=None):
        if x2 is not None:
            x = F.concat([x, x2])
        h = self.c(x)
        if self.noise:
            h = add_noise_c(h)
        if self.bn:
            h = self.b(h)
        if self.activation is not None:
            h = self.activation(h)
        return h
            
gene_base = 64
class Unet(chainer.Chain):
    def __init__(self):
        super(Unet, self).__init__()
        with self.init_scope():
            self.c1=CBR(1,           gene_base*1, mode='down', bn=False)                       #128->64
            self.c2=CBR(gene_base*1, gene_base*2, mode='down')                                 #64->32
            self.c3=CBR(gene_base*2, gene_base*4, mode='down')                                 #32->16
            self.c4=CBR(gene_base*4, gene_base*8, mode='down')                                 #16->8
            self.d4=CBR(gene_base*8, gene_base*4, mode='up')                                   #8->16
            self.d3=CBR(gene_base*8, gene_base*2, mode='up')                                   #16->32
            self.d2=CBR(gene_base*4, gene_base*1, mode='up')                                   #32->64
            self.d1=CBR(gene_base*2, 1,           mode='up',   activation=F.tanh,    bn=False) #64->128

    def __call__(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        y  = self.c4(h3)
        y  = self.d4(y)
        y  = self.d3(y, h3)
        y  = self.d2(y, h2)
        y  = self.d1(y, h1)
        return y

disc_base = 64
class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c1=CBR(1,            gene_base*1,  mode='down', activation=F.leaky_relu, bn=False)  #128->64
            self.c2=CBR(gene_base*1,  gene_base*2,  mode='down', activation=F.leaky_relu)            #64->32
            self.c3=CBR(gene_base*2,  gene_base*4,  mode='down', activation=F.leaky_relu)            #32->16
            self.c4=CBR(gene_base*4,  gene_base*8,  mode='down', activation=F.leaky_relu)            #16->8
            self.lo=L.Linear(None, 1)

    def __call__(self, x):
        x = add_noise_n(x, 0.1)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        y  = self.lo(h4)

        return y, [h4]

