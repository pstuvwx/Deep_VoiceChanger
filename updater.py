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
        self.device = device
        self.iteration = 0
        self.xp = self.generator_ab.xp
        self.bch = iterator_a.batch_size

        # backup_num = 50
        # self.backup_a = self.xp.zeros((backup_num*self.bch, 1, dataset.height, 128), dtype=self.xp.float32)
        # self.backup_b = self.xp.zeros((backup_num*self.bch, 1, dataset.height, 128), dtype=self.xp.float32)
    
    def loss_hinge_disc(self, fake, real):
        loss = F.mean(F.relu(0.1 - real))
        loss += F.mean(F.relu(0.1 + fake))
        return loss

    def loss_hinge_gene(self, fake):
        loss = F.mean(F.relu(-fake))
        return loss

    # def loss_ls_disc(self, fake, real):
    #     loss = F.mean(F.square(0.5 - real))
    #     loss += F.mean(F.square(0.5 + fake))
    #     return loss

    # def loss_ls_gene(self, fake):
    #     loss = F.mean(F.square(fake))
    #     return loss

    # def loss_hinge_mean(self, a, b, eps=0.1):
    #     loss = F.mean(F.relu(F.absolute(a-b)-eps))
    #     return loss

    def clip_weight(self, target, lim):
        for n, c in target.namedparams():
            if 'W' in n:
                m = cp.max(cp.abs(c.data))
                if m > lim:
                    c.data *= lim / m

    def backup_get(self, backup, inp):
        p = np.random.randint(0, backup.shape[0], self.bch)
        q = np.random.randint(0, self.bch, self.bch)
        inp.unchain_backward()
        new = self.xp.copy(inp.data)
        new[q] = backup[p]
        return Variable(new)
    
    def backup_set(self, backup, inp):
        p = np.random.randint(0, backup.shape[0], self.bch)
        q = np.random.randint(0, self.bch, self.bch)
        backup[p] = self.xp.copy(inp.data[q])
    
    def gene_update_half(self, a):
        if a:
            itr_x = self.itr_a
            itr_y = self.itr_b
            gene_xy = self.generator_ab
            gene_yx = self.generator_ba
            disc = self.discriminator_b
            opt = self.opt_g_a
            # backup = self.backup_b
        else:
            itr_x = self.itr_b
            itr_y = self.itr_a
            gene_xy = self.generator_ba
            gene_yx = self.generator_ab
            disc = self.discriminator_a
            opt = self.opt_g_b
            # backup = self.backup_a

        x = Variable(self.converter(itr_x.next(), self.device))
        y = Variable(self.converter(itr_y.next(), self.device))

        xy  = gene_xy(x)
        xyx = gene_yx(xy)
        yy  = gene_xy(y)

        xy_disc = disc(xy)

        recon_loss = F.mean_absolute_error(x, xyx)
        gan_loss   = self.loss_hinge_gene(xy_disc)
        ident_loss = F.mean_absolute_error(y, yy)

        loss_gene = recon_loss*10.0 + gan_loss + ident_loss

        # self.backup_set(backup, xy)

        gene_xy.cleargrads()
        loss_gene.backward()
        opt.update()

        chainer.reporter.report({
            'loss/g/recon': recon_loss,
            'loss/g/ident': ident_loss,
            'loss/g/gene':  gan_loss})
    
    def gene_update_full(self):
        a = Variable(self.converter(self.itr_a.next(), self.device))
        b = Variable(self.converter(self.itr_b.next(), self.device))

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

        loss_gene = recon_loss*10.0 + gan_loss + ident_loss

        # self.backup_set(backup, xy)

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
            # backup = self.backup_a
        else:
            itr_r = self.itr_b
            itr_f = self.itr_a
            gene = self.generator_ab
            disc = self.discriminator_b
            opt = self.opt_d_b
            # backup = self.backup_b

        real = Variable(self.converter(itr_r.next(), self.device))
        fake = Variable(self.converter(itr_f.next(), self.device))
        fake = gene(fake)
        # fake = self.backup_get(backup, fake)

        real = disc(real)
        fake = disc(fake)

        loss_disc = self.loss_hinge_disc(fake, real)

        disc.cleargrads()
        loss_disc.backward()
        opt.update()

        # self.clip_weight(disc, 0.05)

        chainer.reporter.report({
            'loss/d/disc': loss_disc})

    def update_core(self):
        if self.iteration % 2 == 0:
            self.gene_update_full()
        else:
            self.disc_update_half(True)
            self.disc_update_half(False)

