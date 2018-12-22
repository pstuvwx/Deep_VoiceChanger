'''
This is convertor that you can use traind models.
There is two mode: numpy and cupy.
If you don't have GPU or don't installed CUDA, you cannot use cupy_mode.
cupy_mode is very fast than numpy_mode.
'''

import chainer
import dataset
import tqdm
import queue
import numpy as np
from gla.gla_util import GLA
from nets.models import Generator

def numpy_mode():
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            netA_path = input('enter netA path...')
            netB_path = input('enter netB path...')
            wave_path = input('enter wave path...')

            ds = dataset.WaveDataset(wave_path, -1, True)

            netA = Generator()
            netB = Generator()
            chainer.serializers.load_npz(netA_path, netA)
            chainer.serializers.load_npz(netB_path, netB)

            que_a = queue.deque()
            que_ab = queue.deque()
            que_aba = queue.deque()

            gla = GLA()

            print('converting...')
            for i in tqdm.tqdm(range(ds.max//dataset.dif)):
                x_a = ds.get_example(i)
                x_a = chainer.dataset.convert.concat_examples([x_a], -1)
                x_a = chainer.Variable(x_a)

                x_ab = netA(x_a)
                x_aba = netB(x_ab)

                que_a  .append(x_a  .data[0])
                que_ab .append(x_ab .data[0])
                que_aba.append(x_aba.data[0])
            print('done')

            print('phase estimating...')
            for i, que, name in zip(range(3), [que_a, que_ab, que_aba], ['a.wav', 'ab.wav', 'aba.wav']):
                print()
                print(i+1, '/ 3')
                wave   = np.concatenate([gla.inverse(c_f) for i_f in tqdm.tqdm(que) for c_f in dataset.reverse(i_f)])
                print('done...')
                dataset.save(wave_path + name, 16000, wave)
                print('wave-file saved at', wave_path + name)

            print('all done')

def cupy_mode(gpu):
    from gla.gla_gpu import GLA_GPU
    cp = chainer.cuda.cupy

    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            netA_path = input('enter netA path...')
            netB_path = input('enter netB path...')
            wave_path = input('enter wave path...')
            batchsize = int(input('enter batch size...'))
            chainer.cuda.get_device_from_id(gpu).use()

            ds = dataset.WaveDataset(wave_path, -1, True)

            netA = Generator()
            netB = Generator()
            chainer.serializers.load_npz(netA_path, netA)
            chainer.serializers.load_npz(netB_path, netB)
            netA.to_gpu()
            netB.to_gpu()

            que_a = queue.deque()
            que_ab = queue.deque()
            que_aba = queue.deque()

            gla = GLA_GPU(batchsize*4)

            print('converting...')
            l = ds.max//dataset.dif
            for i in tqdm.tqdm(range(0, l, batchsize)):
                x_a = [ds.get_example(_i) for _i in range(i, min([i+batchsize, l]))]
                x_a = chainer.dataset.convert.concat_examples(x_a, gpu)
                x_a = chainer.Variable(x_a)

                x_ab = netA(x_a)
                x_aba = netB(x_ab)

                que_a  .extend([dataset.reverse(_x) for _x in cp.asnumpy(x_a  .data)])
                que_ab .extend([dataset.reverse(_x) for _x in cp.asnumpy(x_ab .data)])
                que_aba.extend([dataset.reverse(_x) for _x in cp.asnumpy(x_aba.data)])
            img_a   = np.concatenate(que_a,   axis=0)
            img_ab  = np.concatenate(que_ab,  axis=0)
            img_aba = np.concatenate(que_aba, axis=0)
            print('done')

            print('phase estimating...')
            for i, img, name in zip(range(3), [img_a, img_ab, img_aba], ['a.wav', 'ab.wav', 'aba.wav']):
                print()
                print(i+1, '/ 3')
                wave = gla.auto_inverse(img)
                print('done...')
                dataset.save(wave_path + name, 16000, wave)
                print('wave-file saved at', wave_path + name)

            print('all done')

if __name__ == "__main__":
    gpu = int(input('enter gpu number(if you have no gpu, enter -1)...'))
    if gpu < 0:
        numpy_mode()
    else:
        cupy_mode(gpu)
   