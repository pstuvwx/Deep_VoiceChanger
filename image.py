import numpy
import os
import random
from PIL import Image


def load(path):
    img = numpy.array(Image.open(path))
    buf = numpy.zeros((img.shape[2], img.shape[0], img.shape[1]), dtype=numpy.uint8)
    for i in range(3):
        buf[i,::] = img[:,:,i]
    img = numpy.float32(buf)
    return img

def save(path, data, rescale=True):
    data = numpy.array(data)
    if rescale:
        data *= 256
        # data += 128
    if data.dtype != numpy.uint8:
        data = numpy.clip(data, 0, 255)
        data = data.astype(numpy.uint8)
    buf = numpy.zeros((data.shape[1], data.shape[2], data.shape[0]), dtype=numpy.uint8)
    for i in range(3):
        a = data[i,:,:]
        buf[:,:,i] = a
    img = Image.fromarray(buf)
    img.save(path)

def save_gray(path, data, rescale=True):
    data = numpy.array(data)[0]
    if rescale:
        data *= 256
        # data += 128
    if data.dtype != numpy.uint8:
        data = data.astype(numpy.uint8)
    img = Image.fromarray(data)
    img.save(path)

def find_file(path):
    name = os.listdir(path)
    dst = [path + "\\" + n for n in name]
    return dst, name

def images_load(src, scale1=True):
    dst = [load(s) for s in src]
    if scale1:
        for d in dst:
            d -= 128
            d /= 128
    return dst
