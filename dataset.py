import numpy as np
import os
import random
import scipy.io.wavfile as wav
import chainer
import pickle

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

height = 128
sride = 64
dif = height*sride

class WaveDataset(chainer.dataset.DatasetMixin):
    def __init__(self, wave, dataset_len, test):
        self.wave = np.array(load(wave)[1], dtype=float)
        self.length = dataset_len
        self.max = len(self.wave)-dif-sride*3
        self.window = np.hanning(254)
        self.test = test

    def __len__(self):
        return self.length
    
    def get_example(self, i):
        if self.test:
            p = i * dif
            return wave2input_image(self.wave, self.window, p)
        else:
            while True:
                p = random.randint(0, self.max)
                if np.max(self.wave[p:p+dif]) > 1000:
                    break
        return wave2input_image(self.wave, self.window, p)

def wave2input_image(wave, window, pos=0):
    wave_image = np.hstack([wave[pos+i*sride:pos+i*sride+dif].reshape(height, sride) for i in range(256//sride)])[:,:254]
    wave_image *= window
    spectrum_image = np.fft.fft(wave_image, axis=1)
    input_image = np.abs(spectrum_image[:,:128].reshape(1, height, 128), dtype=np.float32)

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
    src = output_image[0]
    src[src > 1] = 1
    src *= scale
    src -= bias
    np.abs(src, out=src)
    np.exp(src, out=src)
    
    src[src < 1] = 1

    mil = np.array(src[:,1:127][:,::-1])
    src = np.concatenate([src, mil], 1)

    return src.astype(complex)

if __name__ == "__main__":
    import image
    from gla.gla_util import GLA
    import tqdm
    import matplotlib.pyplot as plt

    path = input('wave path...')
    ds = WaveDataset(path, -1, True)
    num = ds.max // dif
    gla = GLA()
    wave = []

    for i in tqdm.tqdm(range(num)):
        img = ds.get_example(i)
        image.save_gray('img{}.png'.format(i), img)
        rev = reverse(img)
        for r in rev:
            wave.append(gla.inverse(r))
    
    wave = np.concatenate(wave)
    save('wave.wav', 16000, wave)
