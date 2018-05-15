import numpy as np
import os
import random
import scipy.io.wavfile as wav

def load(path, time=-1):
    bps, data = wav.read(path)
    if len(data.shape) != 1:
        data = data[:,0] + data[:,1]
    if time > 0:
        length = int(bps * time)
        if length <= len(data):
            dst = data[0:length]
        else:
            dst = np.zeros(length)
            dst[0:len(data)] = data
        data = dst
    return bps, data

def save(path, bps, data):
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    data = np.reshape(data, -1)
    wav.write(path, bps, data)

def find_wav(path):
    name = os.listdir(path)
    dst = [path + "/" + n for n in name]
    return dst, name

def image_single_split_pad(src, side, pos, power, scale, window):
    wave_len = side*2 - 2
    spl = np.array([src[p:p+wave_len]*window for p in range(pos, pos+side*side, side)])
    spl = np.fft.fft(spl, axis=1)
    spl = spl[:,:side]
    spl = np.abs([spl], dtype=np.float32)
    spl = _pow_scale(spl, power)
    spl *= scale
    return spl

def image_single_pad(src, side, power, scale, window):
    wave_len = side*2-2
    src = np.array(src)
    src *= scale
    src = _pow_scale(src, power)
    src = src.reshape((side, side))
    mil = np.array(src[:,1:side-1][:,::-1])
    src = np.concatenate([src, mil], 1)
    mil = None

    src = FGLA(src, wave_len, side, side, window)
    return src

def pow_scale(fft, p):
    r = fft.real
    i = fft.imag
    r = _pow_scale(r, p)
    i = _pow_scale(i, p)
    return r + i * 1j

def _pow_scale(fft, p):
    return np.power(np.abs(fft), p) * np.sign(fft)

def overwrap(fft, length, dif, side):
    dst = np.zeros(dif * (side-1)+length, dtype=float)
    for i, f in enumerate(fft):
        dst[i*dif:i*dif+length] += np.fft.ifft(f).real
    return dst

def split(w, length, dif, side, window):
    dst = np.array([np.fft.fft(w[i:i+length]*window) for i in range(0, side*dif, dif)])
    return dst

def GLA(fft, length, dif, side, window):
    X = [f for f in fft]
    for i in range(len(fft)):
        s = np.random.randn(fft[i].shape[0])*3.1415926
        X[i] = fft[i] * np.sin(s) * 1j + fft[i] * np.cos(s)
    for _ in range(100):
        x = overwrap(X, length, dif, side)
        X = split(x, length, dif, side, window)
        X = [f * _X / np.abs(_X) for f, _X in zip(fft, X)]
    return overwrap(X, length, dif, side)

def FGLA(fft, length, dif, side, window):
    X = [f for f in fft]
    for i in range(len(fft)):
        s = np.random.randn(fft[i].shape[0])*3.1415926
        X[i] = fft[i] * np.sin(s) * 1j + fft[i] * np.cos(s)
    alpha = 0.99
    for _ in range(100):
        L = X
        x = overwrap(X, length, dif, side)
        X = split(x, length, dif, side, window)
        X = fft * X / np.abs(X)
        X = X + alpha*(X - L)
    return overwrap(X, length, dif, side)
