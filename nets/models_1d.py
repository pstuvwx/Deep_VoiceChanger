import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Sequential
from .sn_linear import SNLinear
from .block_1d import ConvBlock, ResBlock, SNResBlock, SNConvBlock, SNLinearBlock, SNMDBlock

def add_noise(h, sigma=0.1, force=False):
    if chainer.config.train or force:
        xp = chainer.backends.cuda.get_array_module(h.data)
        return h + sigma * xp.random.randn(*h.data.shape, dtype=xp.float32)
    else:
        return h

def x_tanh(x):
    return 0.2*x + F.tanh(x)

def leaky_relu001(x):
    return F.leaky_relu(x, 0.01)

class Generator(chainer.ChainList):
    def __init__(self, base=64):
        super(Generator, self).__init__(
                ConvBlock(1,       base*1,  mode='none'),
                ConvBlock(base*1,  base*2,  mode='down'),
                ConvBlock(base*2,  base*4,  mode='down'),
                ResBlock (base*4,  base*4),
                ResBlock (base*4,  base*4),
                ResBlock (base*4,  base*4),
                ResBlock (base*4,  base*4),
                ResBlock (base*4,  base*4),
                ConvBlock(base*4,  base*2,  mode='up'),
                ConvBlock(base*2,  base*1,  mode='up'),
                ConvBlock(base*1,  1,       mode='none', bn=False, activation=F.sigmoid)
            )
    
    def __call__(self, x):
        b = x.shape[0]
        c = 1#x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        x = F.reshape(x, (b*h,c,w))
        for l in self:
            x = l(x)
        x = F.reshape(x, (b,c,h,w))
        return x
    
class Discriminator(chainer.ChainList):
    def __init__(self, base=64):
        super(Discriminator, self).__init__(
                SNConvBlock(1,       base*1,  mode='none'),
                SNConvBlock(base*1,  base*2,  mode='down'),
                SNConvBlock(base*2,  base*4,  mode='down'),
                SNConvBlock(base*4,  base*8,  mode='down'),
                SNConvBlock(base*8,  base*16, mode='down'),
                SNConvBlock(base*16, base*32, mode='down'),
                SNMDBlock(base*32, in_size=4)
            )
    
    def __call__(self, x):
        b = x.shape[0]
        c = 1#x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        x = F.reshape(x, (b*h,c,w))

        x = add_noise(x)
        for l in self:
            x = l(x)
        return x
    
