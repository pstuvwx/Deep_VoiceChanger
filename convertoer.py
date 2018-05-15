# coding: UTF-8
import argparse
import os
import random

import numpy as np
import cupy as cp
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert
from chainer import cuda

import waver

bps = 16000
side = 128
wave_len = side*2-2
wind = np.hanning(wave_len)
test_len = 10
data_size = 10000
fft_resca = 15
fft_scale = 1/fft_resca
pow_scale = 0.2
    
class CBR(chainer.Chain):
    def __init__(self, in_ch, out_ch, bn=True, activation=F.leaky_relu, mode='down', noise=False, res=None):
        super(CBR, self).__init__()
        with self.init_scope():
            self.bn = bn
            self.activation = activation
            self.noise = noise
            self.res = res
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
        if self.bn:
            h = self.b(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.res is not None:
            h = self.res(h)
        return h
            
gene_base = 64
class Unet(chainer.Chain):
    def __init__(self):
        super(Unet, self).__init__()
        with self.init_scope():
            self.c1=CBR(1,           gene_base*1, mode='down', bn=False) #256->128
            self.c2=CBR(gene_base*1, gene_base*2, mode='down')           #128->64
            self.c3=CBR(gene_base*2, gene_base*4, mode='down')           #64->32
            self.c4=CBR(gene_base*4, gene_base*8, mode='down')           #32->16
            self.d4=CBR(gene_base*8, gene_base*4, mode='up')             #16->32
            self.d3=CBR(gene_base*8, gene_base*2, mode='up')             #32->64
            self.d2=CBR(gene_base*4, gene_base*1, mode='up')             #64->128
            self.d1=CBR(gene_base*2, 1,           mode='up',   activation=F.tanh,    bn=False) #128->256

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

def load_comp(wave, num):
    images = [waver.image_single_split_pad(wave, side, i*side*(side-1)+wave_len, pow_scale, fft_scale, wind) for i in range(num)]
    return images

def save_single(path, bps, src, side, power, scale):
    if power is not None:
        power = 1 / power
    wave = waver.image_single_pad(src, side, power, scale, wind)
    waver.save(path, bps, wave)

def save_comp(path, bps, srces, side, power, scale):
    if power is not None:
        power = 1 / power
    waves = [waver.image_single_pad(s, side, power, scale, wind) for s in srces]
    wave = np.hstack(waves)    
    waver.save(path, bps, wave)

def main():
    path = input("wave path...")

    bps, wave = waver.load(path)

    generator_ab = Unet()
    cp.cuda.Device(0).use()
    generator_ab.to_gpu()

    netpath = input("net path...")
    chainer.serializers.load_npz(netpath, generator_ab)

    with chainer.using_config('train', False):
        batch_a = load_comp(wave, 32)
        x_a = convert.concat_examples(batch_a, 0)
        x_a = chainer.Variable(x_a)

        x_ab = generator_ab(x_a)

        x_a = cp.asnumpy(x_a.data)
        x_ab = cp.asnumpy(x_ab.data)

        save_comp('a.wav',  bps, x_a,  side, pow_scale, fft_resca)
        save_comp('ab.wav', bps, x_ab, side, pow_scale, fft_resca)

if __name__ == '__main__':
    main()
