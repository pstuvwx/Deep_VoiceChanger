import chainer
import dataset
import tqdm
import queue
import numpy as np
from gla.gla_util import GLA
from nets.models import Generator

def main():
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            netA_path = input('netA path...')
            netB_path = input('netB path...')
            wave_path = input('waveA path...')

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
                print(i, '/ 3')
                wave   = np.concatenate([gla.inverse(c_f) for i_f in tqdm.tqdm(que)   for c_f in dataset.reverse(i_f)])
                print('done...')
                dataset.save(wave_path + name, 16000, wave)
                print('wave-file saved at', wave_path + name)

            print('all done')


if __name__ == "__main__":
   main()
   