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
from nets.models import Generator, Discriminator, padding
from dataset import WaveDataset, PreEncodedDataset
from updater import Updater
from gla.gla_util import GLA

def init_gene(gpu):
    n = Generator()
    n.to_gpu()

    o = chainer.optimizers.Adam(5e-4, beta1=0.0, beta2=0.9)
    o.setup(n)
    o.add_hook(chainer.optimizer.GradientClipping(10))
    o.add_hook(chainer.optimizer.WeightDecay(1e-4))

    return n, o
    

def init_disc(gpu):
    n = Discriminator()
    n.to_gpu()

    o = chainer.optimizers.Adam(5e-4, beta1=0.0, beta2=0.9)
    o.setup(n)
    o.add_hook(chainer.optimizer.GradientClipping(10))
    o.add_hook(chainer.optimizer.WeightDecay(1e-4))

    return n, o


def init_dataset(path, length, is_test, batchsize):
    if path[-4:] == '.npy':
        ds = PreEncodedDataset(path, length, is_test)
    elif path[-4:] == '.wav':
        ds =  WaveDataset(path, length, is_test)
    else:
        raise Exception('file format is missing')
    
    return chainer.iterators.SerialIterator(ds, batchsize, shuffle=False)


def resume(args, trainer, generator_ab, generator_ba, discriminator_a, discriminator_b):
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
    

def preview_convert(iterator_a, iterator_b, g_a, g_b, device, gla, dst):
    @chainer.training.make_extension()
    def make_preview(trainer):
        with chainer.using_config('train', False):
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
                    i = cp.asnumpy(i.data)[:,:,padding:-padding,:].reshape(1, -1, 128)
                    image.save(image_dir+'/{}{}.jpg'.format(trainer.updater.epoch,n), i)
                    w = np.concatenate([gla.inverse(_i) for _i in dataset.reverse(i)])
                    dataset.save(preview_dir+'/{}{}.wav'.format(trainer.updater.epoch,n), 16000, w)

    return make_preview


def main():
    parser = argparse.ArgumentParser(description='Deep_VoiceChanger')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=100000,
                        help='Number of to train iteration')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--out', '-o', default='results',
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
                        help='Path of train wave file of voice a')
    parser.add_argument('--voice_b', '-w', default='../src/nekomasu_long.wav',
                        help='Path of train wave file of voice b')
    parser.add_argument('--test_a', '-s', default='../src/KizunaAI_short.wav',
                        help='Path of test wave file of voice a')
    parser.add_argument('--test_b', '-u', default='../src/nekomasu_short.wav',
                        help='Path of test wave file of voice b')
    args = parser.parse_args()

    chainer.cuda.set_max_workspace_size(256*1024*1024)
    chainer.config.type_check = False
    chainer.config.autotune = True 

    if args.gpu < 0:
        print('sorry, but CPU is not recommended')
        quit()
    cp.cuda.Device(args.gpu).use()

    if args.test_a == '':
        args.test_a = args.voice_a
    if args.test_b == '':
        args.test_b = args.voice_b

    generator_ab,    opt_g_a = init_gene(args.gpu)
    generator_ba,    opt_g_b = init_gene(args.gpu)
    discriminator_a, opt_d_a = init_disc(args.gpu)
    discriminator_b, opt_d_b = init_disc(args.gpu)

    gla = GLA()

    train_iter_a = init_dataset(args.voice_a, 20000, False, args.batchsize)
    train_iter_b = init_dataset(args.voice_b, 20000, False, args.batchsize)
    test_iter_a  = init_dataset(args.test_a,  -1,    True,  16)
    test_iter_b  = init_dataset(args.test_b,  -1,    True,  16)

    updater = Updater(train_iter_a, train_iter_b, opt_g_a, opt_g_b, opt_d_a, opt_d_b, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(extensions.snapshot(filename='snapshot.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(generator_ab,    'generator_ab.npz'),    trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(generator_ba,    'generator_ba.npz'),    trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(discriminator_a, 'discriminator_a.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(discriminator_b, 'discriminator_b.npz'), trigger=(10, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'loss/g/recon', 'loss/g/ident', 'loss/g/gene', 'loss/d/disc', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(extensions.ExponentialShift("alpha", 0.1, optimizer=opt_g_a), trigger=(25000, 'iteration'))
    trainer.extend(extensions.ExponentialShift("alpha", 0.1, optimizer=opt_g_b), trigger=(25000, 'iteration'))
    trainer.extend(extensions.ExponentialShift("alpha", 0.1, optimizer=opt_d_a), trigger=(25000, 'iteration'))
    trainer.extend(extensions.ExponentialShift("alpha", 0.1, optimizer=opt_d_b), trigger=(25000, 'iteration'))
    trainer.extend(preview_convert(test_iter_a, test_iter_b, generator_ab, generator_ba, args.gpu, gla, args.out), trigger=(1, 'epoch'))

    resume(args, trainer, generator_ab, generator_ba, discriminator_a, discriminator_b)

    trainer.run()

if __name__ == '__main__':
    main()
