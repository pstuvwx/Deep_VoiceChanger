import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import convert
from chainer import Variable
import dataset
import random

class Updater(training.StandardUpdater):
    def __init__(self, iterator_a, iterator_b, opt_g_a, opt_g_b, opt_d_a, opt_d_b, device):
        self._iterators = {'main': iterator_a, 'second': iterator_b}
        self.generator_ab = opt_g_a.target
        self.generator_ba = opt_g_b.target
        self.discriminator_a = opt_d_a.target
        self.discriminator_b = opt_d_b.target
        self._optimizers = {
            'generator_ab': opt_g_a,
            'generator_ba': opt_g_b,
            'discriminator_a': opt_d_a,
            'discriminator_b': opt_d_b,
            }
           
        self.itr_a = iterator_a
        self.itr_b = iterator_b
        self.opt_g_a = opt_g_a
        self.opt_g_b = opt_g_b
        self.opt_d_a = opt_d_a
        self.opt_d_b = opt_d_b

        self.converter = convert.concat_examples
        self._device = device
        self.iteration = 0
        self.xp = self.generator_ab.xp
        self.bch = iterator_a.batch_size

    def loss_hinge_disc(self, fake, real):
        loss = F.mean(F.relu(0.5 - real))
        loss += F.mean(F.relu(0.5 + fake))
        return loss

    def loss_hinge_gene(self, fake):
        loss = F.mean(F.relu(-fake))
        return loss

    def gene_update_half(self, a):
        if a:
            itr_x = self.itr_a
            itr_y = self.itr_b
            gene_xy = self.generator_ab
            gene_yx = self.generator_ba
            disc = self.discriminator_b
            opt = self.opt_g_a
        else:
            itr_x = self.itr_b
            itr_y = self.itr_a
            gene_xy = self.generator_ba
            gene_yx = self.generator_ab
            disc = self.discriminator_a
            opt = self.opt_g_b

        x = Variable(self.converter(itr_x.next(), self._device))
        y = Variable(self.converter(itr_y.next(), self._device))

        xy  = gene_xy(x)
        xyx = gene_yx(xy)
        yy  = gene_xy(y)

        xy_disc = disc(xy)

        recon_loss = F.mean_absolute_error(x, xyx)
        gan_loss   = self.loss_hinge_gene(xy_disc)
        ident_loss = F.mean_absolute_error(y, yy)

        loss_gene = recon_loss*3.0 + gan_loss + ident_loss*0.5

        gene_xy.cleargrads()
        loss_gene.backward()
        opt.update()

        chainer.reporter.report({
            'loss/g/recon': recon_loss,
            'loss/g/ident': ident_loss,
            'loss/g/gene':  gan_loss})
    
    def gene_update_full(self):
        a = Variable(self.converter(self.itr_a.next(), self._device))
        b = Variable(self.converter(self.itr_b.next(), self._device))

        ab  = self.generator_ab(a)
        ba  = self.generator_ba(b)
        aba = self.generator_ba(ab)
        bab = self.generator_ab(ba)
        aa  = self.generator_ba(a)
        bb  = self.generator_ab(b)

        ab_disc = self.discriminator_b(ab)
        ba_disc = self.discriminator_a(ba)

        recon_loss = F.mean_absolute_error(a, aba) + F.mean_absolute_error(b, bab)
        gan_loss   = self.loss_hinge_gene(ab_disc) + self.loss_hinge_gene(ba_disc)
        ident_loss = F.mean_absolute_error(a, aa)  + F.mean_absolute_error(b, bb)

        loss_gene = recon_loss*3.0 + gan_loss + ident_loss*0.5

        self.generator_ab.cleargrads()
        self.generator_ba.cleargrads()
        loss_gene.backward()
        self.opt_g_a.update()
        self.opt_g_b.update()

        chainer.reporter.report({
            'loss/g/recon': recon_loss,
            'loss/g/ident': ident_loss,
            'loss/g/gene':  gan_loss})
    
    def disc_update_half(self, a):
        if a:
            itr_r = self.itr_a
            itr_f = self.itr_b
            gene = self.generator_ba
            disc = self.discriminator_a
            opt = self.opt_d_a
        else:
            itr_r = self.itr_b
            itr_f = self.itr_a
            gene = self.generator_ab
            disc = self.discriminator_b
            opt = self.opt_d_b

        real = Variable(self.converter(itr_r.next(), self._device))
        fake = Variable(self.converter(itr_f.next(), self._device))
        fake = gene(fake)

        real = disc(real)
        fake = disc(fake)

        loss_disc = self.loss_hinge_disc(fake, real)

        disc.cleargrads()
        loss_disc.backward()
        opt.update()

        chainer.reporter.report({
            'loss/d/disc': loss_disc})

    def update_core(self):
        if self.iteration % 2 == 0:
            self.gene_update_half(True)
            self.gene_update_half(False)
        else:
            self.disc_update_half(True)
            self.disc_update_half(False)

