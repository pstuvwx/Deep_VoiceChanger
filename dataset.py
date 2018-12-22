'''
This code is about dataset.
There is two dataset in here. You can chose both dataset.

WaveDataset takes a wave file.
This dataset convert from a wave file to input datas.
The process what has many times FFT is repeated, so this use many CPU resouces.

PreEncodedDataset takes a numpy file that is pre-encoded datas.
You can get the pre-encoded file by runnning this .py file, or pre_encode method.
'''

import numpy as np
import os
import random
import scipy.io.wavfile as wav
import chainer
import pickle
from nets.models import padding

def load(path):
    bps, data = wav.read(path)
    if len(data.shape) != 1:
        data = data[:,0] + data[:,1]
    return bps, data

def save(path, bps, data):
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    data = np.reshape(data, -1)
    wav.write(path, bps, data)

def find_wav(path):
    name = os.listdir(path)
    dst = []
    for n in name:
        if n[-4:] == '.wav':
            dst.append(path + "/" + n)
    return dst

scale = 9
bias = -6.2

height = 64
sride = 64
dif = height*sride

class WaveDataset(chainer.dataset.DatasetMixin):
    def __init__(self, wave, dataset_len, test):
        self.wave = np.array(load(wave)[1], dtype=float)
        self.max = len(self.wave)-dif-sride*(3+padding*2)
        self.length = dataset_len
        if dataset_len <= 0:
            self.length = self.max // dif
        self.window = np.hanning(254)
        self.test = test

    def __len__(self):
        return self.length
    
    def get_example(self, i):
        if self.test:
            p = i * dif
        else:
            while True:
                p = random.randint(0, self.max)
                if np.max(self.wave[p:p+dif]) > 1000:
                    break
        return wave2input_image(self.wave, self.window, p, padding)

class PreEncodedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, encoded_path, dataset_len, test):
        self.images = np.load(encoded_path)
        self.max = self.images.shape[1]-height - padding*2
        self.length = dataset_len
        if dataset_len <= 0:
            self.length = self.max // height
        self.test = test

    def __len__(self):
        return self.length
    
    def get_example(self, i):
        if self.test:
            p = i * height
        else:
            while True:
                p = random.randint(0, self.max)
                if np.max(self.images[:,p:p+height,:]) > 0.4:
                    break
        return np.copy(self.images[:,p:p+height+padding*2,:])

def wave2input_image(wave, window, pos=0, pad=0):
    wave_image = np.hstack([wave[pos+i*sride:pos+(i+pad*2)*sride+dif].reshape(height+pad*2, sride) for i in range(256//sride)])[:,:254]
    wave_image *= window
    spectrum_image = np.fft.fft(wave_image, axis=1)
    input_image = np.abs(spectrum_image[:,:128].reshape(1, height+pad*2, 128), dtype=np.float32)

    np.clip(input_image, 1000, None, out=input_image)
    np.log(input_image, out=input_image)
    input_image += bias
    input_image /= scale

    if np.max(input_image) > 0.95:
        print('input image max bigger than 0.95', np.max(input_image))
    if np.min(input_image) < 0.05:
        print('input image min smaller than 0.05', np.min(input_image))

    return input_image

def reverse(output_image):
    src = output_image[0,padding:-padding,:]
    src[src > 1] = 1
    src *= scale
    src -= bias
    np.abs(src, out=src)
    np.exp(src, out=src)
    
    src[src < 1000] = 1

    mil = np.array(src[:,1:127][:,::-1])
    src = np.concatenate([src, mil], 1)

    return src.astype(complex)


def pre_encode():
    import tqdm

    path = input('enter wave path...')
    ds = WaveDataset(path, -1, True)
    num = ds.max // dif

    imgs = [ds.get_example(i) for i in tqdm.tqdm(range(num))]    
    dst = np.concatenate(imgs, axis=1)
    print(dst.shape)

    np.save(path[:-3]+'npy', dst)
    print('encoded file saved at', path[:-3]+'npy')


if __name__ == "__main__":
    pre_encode()
