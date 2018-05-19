import argparse
import os
import random

import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert
from chainer import cuda

import waver
from model import Unet, Discriminator, add_noise_n, gene_base, disc_base, ResUnet


bps = 16000
side = 128
wave_len = side*2-2
wind = np.hanning(wave_len)
test_len = 10
data_size = 10000
fft_resca = 18
fft_scale = 1/fft_resca
pow_scale = 0.2
    
class WaverDataset(chainer.dataset.DatasetMixin):
    def __init__(self, wave, length):
        self.wave = np.array(wave)
        self.length = length
        self.max = len(wave)-side*(side-1)-wave_len

    def __len__(self):
        return self.length

    def get_example(self, i):
        p = random.randint(0, self.max)
        image = waver.image_single_split_pad(self.wave, side, p, pow_scale, fft_scale, wind)
        if np.max(image) > 1:
            print(np.max(image))
            print("max value of data exceeded 1!")
            print("you should change \'fft_resca\'")
        return image

class TestWaverDataset(chainer.dataset.DatasetMixin):
    def __init__(self, wave, length):
        self.wave = np.array(wave)
        self.length = length
        self.max = len(wave)-side*(side-1)-wave_len
        if test_len*(side*(side-1)+wave_len) > self.max:
            print("wave file length is too short!")
            print("you have to use more long wave file than 11 second.(and 16kHz sampling rate)")

    def __len__(self):
        return self.length

    def get_example(self, i):
        p = i * (side*(side-1)+wave_len)
        image = waver.image_single_split_pad(self.wave, side, p, pow_scale, fft_scale, wind)
        return image


class DiscoGANUpdater(training.StandardUpdater):
    def __init__(self, iterator_a, iterator_b, opt_g_ab, opt_g_ba, opt_d_a, opt_d_b, device):
        self._iterators = {'main': iterator_a, 'second': iterator_b}
        self.generator_ab = opt_g_ab.target
        self.generator_ba = opt_g_ba.target
        self.discriminator_a = opt_d_a.target
        self.discriminator_b = opt_d_b.target
        self._optimizers = {'generator_ab': opt_g_ab,
                            'generator_ba': opt_g_ba,
                            'discriminator_a': opt_d_a,
                            'discriminator_b': opt_d_b}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0
        self.xp = self.generator_ab.xp

    def real_ans(self, bch):
        return self.xp.ones((bch, 1, 1, 1), dtype=cp.float32)*0.9

    def fake_ans(self, bch):
        return self.xp.ones((bch, 1, 1, 1), dtype=cp.float32)*0.1

    def compute_loss_gan(self, y_real, y_fake):
        batchsize = y_real.shape[0]
        loss_dis = F.sum(F.softplus(-y_real) + F.softplus(y_fake))
        loss_gen = F.sum(F.softplus(-y_fake))
        return loss_dis / batchsize, loss_gen / batchsize
        # loss_dis = F.mean_squared_error(y_real, self.real_ans(batchsize))*0.5 + F.mean_squared_error(y_fake, self.fake_ans(batchsize))*0.5
        # loss_gen = F.mean_squared_error(y_fake, self.real_ans(batchsize))
        # return loss_dis, loss_gen

    def compute_loss_feat(self, feats_real, feats_fake):
        losses = 0
        for feat_real, feat_fake in zip(feats_real, feats_fake):
            feat_real_mean = F.sum(feat_real, 0) / feat_real.shape[0]
            feat_fake_mean = F.sum(feat_fake, 0) / feat_fake.shape[0]
            l2 = (feat_real_mean - feat_fake_mean) ** 2
            loss = F.sum(l2) / l2.size
            losses += loss
        return losses

    def update_core(self):
        batch_a = self._iterators['main'].next()
        x_a = self.converter(batch_a, self.device)

        batch_b = self._iterators['second'].next()
        x_b = self.converter(batch_b, self.device)

        x_ab = self.generator_ab(x_a)
        x_ba = self.generator_ba(x_b)

        x_aba = self.generator_ba(x_ab)
        x_bab = self.generator_ab(x_ba)

        recon_loss_a = F.mean_squared_error(x_a, x_aba)
        recon_loss_b = F.mean_squared_error(x_b, x_bab)

        y_a_real, feats_a_real = self.discriminator_a(x_a)
        y_a_fake, feats_a_fake = self.discriminator_a(x_ba)

        y_b_real, feats_b_real = self.discriminator_b(x_b)
        y_b_fake, feats_b_fake = self.discriminator_b(x_ab)

        gan_loss_dis_a, gan_loss_gen_a = self.compute_loss_gan(y_a_real, y_a_fake)
        feat_loss_a = self.compute_loss_feat(feats_a_real, feats_a_fake)

        gan_loss_dis_b, gan_loss_gen_b = self.compute_loss_gan(y_b_real, y_b_fake)
        feat_loss_b = self.compute_loss_feat(feats_b_real, feats_b_fake)

        gan = 0.1
        feat = 0.9
        recon = 1

        total_loss_gen_a = gan*gan_loss_gen_b + feat*feat_loss_b + recon*recon_loss_a
        total_loss_gen_b = gan*gan_loss_gen_a + feat*feat_loss_a + recon*recon_loss_b

        gen_loss = total_loss_gen_a + total_loss_gen_b
        dis_loss = gan_loss_dis_a + gan_loss_dis_b

        if self.iteration % 3 != 0:
            self.generator_ab.cleargrads()
            self.generator_ba.cleargrads()
            gen_loss.backward()
            self._optimizers['generator_ab'].update()
            self._optimizers['generator_ba'].update()
        else:
            wei_max = 0.01
            self.discriminator_a.cleargrads()
            self.discriminator_b.cleargrads()
            dis_loss.backward()
            self._optimizers['discriminator_a'].update()
            self._optimizers['discriminator_b'].update()
            for n, l in self.discriminator_a.namedparams():
                if 'W' in n:
                    m = cp.max(l.data)
                    if m > wei_max:
                        l.data *= wei_max / m
            for n, l in self.discriminator_b.namedparams():
                if 'W' in n:
                    m = cp.max(l.data)
                    if m > wei_max:
                        l.data *= wei_max / m

        chainer.reporter.report({
            'loss/gene': gen_loss.data,
            'loss/gan': gan_loss_gen_a.data + gan_loss_gen_b.data,
            'loss/feat': feat_loss_a.data + feat_loss_b.data,
            'loss/recon': recon_loss_a.data + recon_loss_b.data,
            'rave':(F.average(y_a_real).data+F.average(y_b_real).data)*0.5,
            'fave':(F.average(y_a_fake).data+F.average(y_b_fake).data)*0.5,
            'loss/disc': dis_loss.data})

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
    parser = argparse.ArgumentParser(description='Voice_DiscoGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (CPU is not recommend)')
    parser.add_argument('--n_thread', '-t', type=int, default=8,
                        help='Number of parallel data loading thread')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--voice_a', '-v', default='src/kizunaAI.wav',
                        help='Path of source wave file of voice a')
    parser.add_argument('--voice_b', '-w', default='src/nekomasu.wav',
                        help='Path of source wave file of voice b')
    args = parser.parse_args()

    generator_ab = Unet()
    generator_ba = Unet()
    discriminator_a = Discriminator()
    discriminator_b = Discriminator()
    if args.gpu >= 0:
        cp.cuda.Device(args.gpu).use()
        generator_ab.to_gpu()
        generator_ba.to_gpu()
        discriminator_a.to_gpu()
        discriminator_b.to_gpu()
    else:
        print("I'm sorry.")
        print("Learning with CPU is too slow, GPU is mandatory.")
        quit()

    opt_g_ab = chainer.optimizers.RMSprop(1e-5)
    opt_g_ab.setup(generator_ab)
    opt_g_ba = chainer.optimizers.RMSprop(1e-5)
    opt_g_ba.setup(generator_ba)

    opt_d_a = chainer.optimizers.RMSprop(1e-5)
    opt_d_a.setup(discriminator_a)
    opt_d_a.add_hook(chainer.optimizer.WeightDecay(1e-4))
    opt_d_b = chainer.optimizers.RMSprop(1e-5)
    opt_d_b.setup(discriminator_b)
    opt_d_b.add_hook(chainer.optimizer.WeightDecay(1e-4))


    wave_sourceA = args.voice_a
    wave_sourceB = args.voice_b
    wave_destruction = "{}_unet_{}_{}".format(side, gene_base, disc_base)

    wave_a = TestWaverDataset(waver.load(wave_sourceA)[1], test_len)
    wave_b = TestWaverDataset(waver.load(wave_sourceB)[1], test_len)

    train_a = WaverDataset(waver.load(wave_sourceA)[1], data_size)
    train_b = WaverDataset(waver.load(wave_sourceB)[1], data_size)

    train_iter_a = chainer.iterators.MultithreadIterator(train_a, args.batchsize, shuffle=False, n_threads=args.n_thread)
    train_iter_b = chainer.iterators.MultithreadIterator(train_b, args.batchsize, shuffle=False, n_threads=args.n_thread)
    valid_iter_a = chainer.iterators.SerialIterator(wave_a, test_len, shuffle=False)
    valid_iter_b = chainer.iterators.SerialIterator(wave_b, test_len, shuffle=False)

    updater = DiscoGANUpdater(train_iter_a, train_iter_b, opt_g_ab, opt_g_ba, opt_d_a, opt_d_b, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=wave_destruction+'/'+args.out)

    def out_generated_image(iterator_a, iterator_b, generator_ab, generator_ba, device, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            with chainer.using_config('train', False):
                batch_a = iterator_a.next()
                x_a = convert.concat_examples(batch_a, device)
                x_a = chainer.Variable(x_a)

                batch_b = iterator_b.next()
                x_b = convert.concat_examples(batch_b, device)
                x_b = chainer.Variable(x_b)

                x_ab = generator_ab(x_a)
                x_ba = generator_ba(x_b)

                # x_a = cp.asnumpy(x_a.data)
                # x_b = cp.asnumpy(x_b.data)
                x_ab = cp.asnumpy(x_ab.data)
                x_ba = cp.asnumpy(x_ba.data)

                preview_dir = '{}/preview'.format(dst)
                if not os.path.exists(preview_dir):
                    os.makedirs(preview_dir)
                # save_comp(preview_dir + '/{}a.wav' .format(trainer.updater.epoch), bps, x_a,  side, pow_scale, fft_resca)
                # save_comp(preview_dir + '/{}b.wav' .format(trainer.updater.epoch), bps, x_b,  side, pow_scale, fft_resca)
                save_comp(preview_dir + '/{}ab.wav'.format(trainer.updater.epoch), bps, x_ab, side, pow_scale, fft_resca)
                save_comp(preview_dir + '/{}ba.wav'.format(trainer.updater.epoch), bps, x_ba, side, pow_scale, fft_resca)
        return make_image

    trainer.extend(extensions.snapshot(filename='snapshot'))
    trainer.extend(extensions.snapshot_object(generator_ab, 'generator_ab.npz'))
    trainer.extend(extensions.snapshot_object(generator_ba, 'generator_ba.npz'))
    trainer.extend(extensions.snapshot_object(discriminator_a, 'discriminator_a.npz'))
    trainer.extend(extensions.snapshot_object(discriminator_b, 'discriminator_b.npz'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'loss/gene', 'loss/gan', 'loss/feat', 'loss/recon', 'loss/disc', 'rave', 'fave', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(out_generated_image(valid_iter_a, valid_iter_b, generator_ab, generator_ba, args.gpu, wave_destruction), trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
