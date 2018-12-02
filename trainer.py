import argparse
import os
import numpy as np
import cupy as cp
import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert
from chainer import cuda, Variable

import dataset
import image
from nets.models import Generator, Discriminator
from dataset import WaveDataset
from updater import Updater
from gla.gla_util import GLA

from distutils.dir_util import copy_tree

def init_gene(gpu):
    if gpu < 0:
        print('sorry, but CPU is not recommended')
        quit()
    cp.cuda.Device(gpu).use()

    nets = [Generator(), Generator()]
    opts = []
    for n in nets:
        n.to_gpu()
        o = chainer.optimizers.Adam(1e-5, beta1=0.5, beta2=0.999)
        o.setup(n)
        opts.append(o)

    return nets[0], nets[1], opts[0], opts[1]
    
def init_disc(gpu):
    if gpu < 0:
        print('sorry, but CPU is not recommended')
        quit()
    cp.cuda.Device(gpu).use()

    nets = [Discriminator(), Discriminator()]
    opts = []
    for n in nets:
        n.to_gpu()
        o = chainer.optimizers.Adam(4e-4, beta1=0.5, beta2=0.999)
        o.setup(n)
        opts.append(o)

    return nets[0], nets[1], opts[0], opts[1]
    
def preview_convert(iterator_a, iterator_b, g_a, g_b, device, gla, dst):
    @chainer.training.make_extension()
    def make_preview(trainer):
        with chainer.using_config('train', True):
            with chainer.no_backprop_mode():
                x_a = iterator_a.next()
                x_a = convert.concat_examples(x_a, device)
                x_a = chainer.Variable(x_a)

                x_b = iterator_b.next()
                x_b = convert.concat_examples(x_b, device)
                x_b = chainer.Variable(x_b)

                x_ab = g_a(x_a)
                x_ba = g_b(x_b)

                x_bab = g_a(x_ba)
                x_aba = g_b(x_ab)

                preview_dir = '{}/preview'.format(dst)
                if not os.path.exists(preview_dir):
                    os.makedirs(preview_dir)
                image_dir = '{}/image'.format(dst)
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)

                names = ['a', 'ab', 'aba', 'b', 'ba', 'bab']
                images = [x_a, x_ab, x_aba, x_b, x_ba, x_bab]
                for n, i in zip(names, images):
                    i = cp.asnumpy(i.data).reshape(1, -1, 128)
                    image.save_gray(image_dir+'/{}{}.jpg'.format(trainer.updater.epoch,n), i)
                    w = np.concatenate([gla.inverse(_i) for _i in dataset.reverse(i)])
                    dataset.save(preview_dir+'/{}{}.wav'.format(trainer.updater.epoch,n), 16000, w)

    return make_preview

def main():
    parser = argparse.ArgumentParser(description='Deep_VoiceChanger')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=200000,
                        help='Number of to train iteration')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--n_thread', '-t', type=int, default=2,
                        help='Number of parallel data loading thread')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--gene_ab', '-j', default='',
                        help='Resume generator a2b from file')
    parser.add_argument('--gene_ba', '-k', default='',
                        help='Resume generator b2a from file')
    parser.add_argument('--disc_a', '-m', default='',
                        help='Resume discriminator a from file')
    parser.add_argument('--disc_b', '-l', default='',
                        help='Resume discriminator b from file')
    parser.add_argument('--folder', '-f', default='',
                        help='Resume all model from foledr')
    parser.add_argument('--voice_a', '-v', default='../src/KizunaAI_long.wav',
                        help='Path of source wave file of voice a')
    parser.add_argument('--voice_b', '-w', default='../src/nekomasu_long.wav',
                        help='Path of source wave file of voice b')
    parser.add_argument('--test_a', '-s', default='../src/KizunaAI_short.wav',
                        help='Path of test wave file of voice a')
    parser.add_argument('--test_b', '-u', default='../src/nekomasu_short.wav',
                        help='Path of test wave file of voice b')
    args = parser.parse_args()

    chainer.global_config.autotune = True 

    wave_destruction = "../results/result_128x128_C2R7D05_SN_GAP_leakyrelu02_sigmoid_3_1_1_SNMD1005lr1e4_base32_1on2_half_hinge_mae_batch32_Adam1e5050999_Adam4e4050999"

    generator_ab, generator_ba, opt_g_a, opt_g_b = init_gene(args.gpu)
    discriminator_a, discriminator_b, opt_d_a, opt_d_b = init_disc(args.gpu)
    gla = GLA()

    train_a =  WaveDataset(args.voice_a, 10000, False)
    train_b =  WaveDataset(args.voice_b, 10000, False)
    test_a =   WaveDataset(args.test_a,  100, True)
    test_b =   WaveDataset(args.test_b,  100, True)

    train_iter_a = chainer.iterators.MultithreadIterator(train_a, args.batchsize, shuffle=False, n_threads=args.n_thread)
    train_iter_b = chainer.iterators.MultithreadIterator(train_b, args.batchsize, shuffle=False, n_threads=args.n_thread)
    test_iter_a = chainer.iterators.SerialIterator(test_a, 8, shuffle=False)
    test_iter_b = chainer.iterators.SerialIterator(test_b, 8, shuffle=False)

    updater = Updater(train_iter_a, train_iter_b, opt_g_a, opt_g_b, opt_d_a, opt_d_b, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=wave_destruction+'/'+args.out)

    trainer.extend(extensions.snapshot(filename='snapshot.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(generator_ab, 'generator_ab.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(generator_ba, 'generator_ba.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(discriminator_a, 'discriminator_a.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(discriminator_b, 'discriminator_b.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'loss/g/recon', 'loss/g/ident', 'loss/g/gene', 'loss/d/disc', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(preview_convert(test_iter_a, test_iter_b, generator_ab, generator_ba, args.gpu, gla, wave_destruction), trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    if args.gene_ab:
        chainer.serializers.load_npz(args.gene_ab, generator_ab)
    if args.gene_ba:
        chainer.serializers.load_npz(args.gene_ba, generator_ba)
    if args.disc_a:
        chainer.serializers.load_npz(args.disc_a, discriminator_a)
    if args.disc_b:
        chainer.serializers.load_npz(args.disc_b, discriminator_b)
    if args.folder:
        folder = args.folder
        chainer.serializers.load_npz(folder+'/generator_ab.npz', generator_ab)
        chainer.serializers.load_npz(folder+'/generator_ba.npz', generator_ba)
        chainer.serializers.load_npz(folder+'/discriminator_a.npz', discriminator_a)
        chainer.serializers.load_npz(folder+'/discriminator_b.npz', discriminator_b)
    trainer.run()

if __name__ == '__main__':
    main()
