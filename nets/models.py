import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Sequential
from .sn_linear import SNLinear
from .block import ConvBlock, ResBlock, SNResBlock, SNConvBlock, SNLinearBlock, SNMDBlock, SNL1DBlock, SNCLBlock, CLBlock

padding = 8

def add_noise(h, sigma=0.1, force=False):
    if chainer.config.train or force:
        xp = chainer.backends.cuda.get_array_module(h.data)
        return h + sigma * xp.random.randn(*h.data.shape, dtype=xp.float32)
    else:
        return h

class Generator(chainer.ChainList):
    def __init__(self, base=64):
        super(Generator, self).__init__(
                CLBlock  (1,      base*1, 128),
                CLBlock  (base*1, base*1, 128),
                CLBlock  (base*1, base*1, 128),
                CLBlock  (base*1, base*1, 128),
                CLBlock  (base*1, base*1, 128),
                CLBlock  (base*1, base*1, 128),
                ConvBlock(base*1, 1,       mode='none', activation=F.sigmoid, bn=False)
            )
    
    def __call__(self, x):
        for l in self:
            x = l(x)
        return x
    
class Discriminator(chainer.ChainList):
    def __init__(self, base=64):
        super(Discriminator, self).__init__(
                SNConvBlock(1,       base*1,  mode='down'),
                SNConvBlock(base*1,  base*2,  mode='down'),
                SNConvBlock(base*2,  base*4,  mode='down'),
                SNConvBlock(base*4,  base*8,  mode='down'),
                SNConvBlock(base*8,  base*16, mode='down'),
                SNMDBlock  (base*16, in_size=None, gap=True)
            )
    
    def __call__(self, x):
        x = x[:,:,padding:-padding]
        for l in self:
            x = l(x)
        return x
    
