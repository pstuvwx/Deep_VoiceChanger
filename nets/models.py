import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Sequential
from .sn_linear import SNLinear
from .block import ConvBlock, ResBlock, SNResBlock, FrqBlock, SNFrqBlock, CoPSBlock, SNConvBlock, SNLinearBlock, SNMDBlock

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
    def __init__(self, base=32):
        super(Generator, self).__init__(
                SNConvBlock(1,       base*1,  mode='none'),
                SNConvBlock(base*1,  base*2,  mode='down'),
                SNConvBlock(base*2,  base*4,  mode='down'),
                SNConvBlock(base*4,  base*8,  mode='down'),
                SNConvBlock(base*8,  base*16, mode='down'),
                SNConvBlock(base*16, base*32, mode='down'),
                SNConvBlock(base*32, base*16, mode='up', dr=0.5),
                SNConvBlock(base*16, base*8,  mode='up', dr=0.5),
                SNConvBlock(base*8,  base*4,  mode='up'),
                SNConvBlock(base*4,  base*2,  mode='up'),
                SNConvBlock(base*2,  base*1,  mode='up'),
                SNResBlock (base*1,  base*1,  dr=0.5),
                SNResBlock (base*1,  base*1,  dr=0.5),
                SNResBlock (base*1,  base*1,  dr=0.5),
                ConvBlock  (base*1,  1,       mode='none', activation=None, bn=False)
            )
    
    def __call__(self, x):
        s = []
        for l in self[:6]:
            x = l(x)
            s.append(x)
        x = self[6](x)
        for l, _s in zip(self[7:12], s[:-1][::-1]):
            x = l(x + _s)
        for l in self[12:]:
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
                SNMDBlock(base*16, in_size=None, gap=True)
            )
    
    def __call__(self, x):
        for l in self:
            x = l(x)
        return x
    
