#!/usr/bin/env python
#title           :imageProcess.py
#description     :image process of original image to show which preprocessing are helpful to increase detection accuracy.
#author          :siyu zhu
#date            :April 30rd, 2015
#version         :0.1
#usage           :python imageProcess.py
#notes           :
#python_version  :2.7

import os
import numpy
from fileOp import make_dir
from imgOp import image_load, image_save, unsharp
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2lab, rgb2grey 
from skimage.filters import sobel 
from skimage.feature import canny

indir = '../icdar2013/task21_22/test/image/'
dirlist = os.listdir(indir)

edgedir = '../data/example/edge'
cedgedir = '../data/example/coloredge'
cannydir = '../data/example/cannyedge'
unsharpdir = '../data/example/unsharp'
bilateraldir = '../data/example/bilateral'

make_dir(edgedir)
make_dir(cedgedir)
make_dir(cannydir)
make_dir(unsharpdir)
make_dir(bilateraldir)

for n, f in enumerate(dirlist):
        if not f.endswith('.jpg'):
                continue
        rgbim = image_load(f, indir, flatten = False)
        labim = rgb2lab(rgbim)
        greyim = rgb2grey(rgbim)
        edgeim = sobel(greyim)
        cedgeim = labim 
        cedgeim = numpy.zeros(greyim.shape)
        cedgeim =  cedgeim + sobel(labim[:, :, 0])
        cedgeim =  cedgeim + sobel(labim[:, :, 1])
        cedgeim =  cedgeim + sobel(labim[:, :, 2])
        cannyim = canny(greyim)
        unsharpim = unsharp(rgbim)
        bilateralim = denoise_bilateral(rgbim, sigma_range = 0.5, win_size = 10)
        image_save(edgeim, f, edgedir)
        image_save(cedgeim, f, cedgedir)
        image_save(cannyim, f, cannydir)
        image_save(unsharpim, f, unsharpdir)
        image_save(bilateralim, f, bilateraldir)
        print 'file name', f,
        print 'image shape', rgbim.shape,
        print 'No. of image', n, 'in', len(dirlist)


