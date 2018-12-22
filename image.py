import numpy as np
import os
import random
from PIL import Image


def load(path):
    return PIL2Chainer(Image.open(path))

def save(path, data, rescale=True):
    Chainer2PIL(data, rescale).save(path)

def PIL2Chainer(img, scale=True):
    img = np.array(img)
    if len(img.shape) == 2:
        img = img.astype(np.float32).reshape((1, img.shape[0], img.shape[1]))
    else:
        buf = np.zeros((img.shape[2], img.shape[0], img.shape[1]), dtype=np.uint8)
        for i in range(3):
            buf[i,::] = img[:,:,i]
        img = buf.astype(np.float32)
    if scale:
        # img -= 128
        img /= 256
    return img

def Chainer2PIL(data, rescale=True):
    data = np.array(data)
    if rescale:
        data *= 256
        # data += 128
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255)
        data = data.astype(np.uint8)
    if data.shape[0] == 1:
        buf = data.astype(np.uint8).reshape((data.shape[1], data.shape[2]))
    else:
        buf = np.zeros((data.shape[1], data.shape[2], data.shape[0]), dtype=np.uint8)
        for i in range(3):
            a = data[i,:,:]
            buf[:,:,i] = a
    img = Image.fromarray(buf)
    return img

