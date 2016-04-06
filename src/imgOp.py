#!/usr/bin/env python
#title           :imgOp.py
#description     :functions for image operations
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage		 : from imgOp import * 
#notes           :
#python_version  :2.7

# import modules
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')

import os
import numpy
import scipy
import cv2

from bbOp import xywh2xy, coor_resize
from sys import argv
from scipy.misc import imread, imsave, imresize
from skimage.measure import label as skimage_label
from skimage.filters import gaussian_filter, sobel
from skimage.draw import line as skimage_line
from skimage.draw import set_color as skimage_set_color
from skimage.feature import canny
from sys import maxint
from scipy.signal import convolve2d as conv2
from skimage.morphology import dilation

def image_load(filename, dirname, flatten = True):
	'''_image_load(filename, dirname, flattern = True)

	load image
	Return 2D_numpy_array containing image'''
        if flatten == True:
                img = imread(os.path.join(dirname, filename), flatten = True)
                return numpy.transpose(img)
        else:
                img = imread(os.path.join(dirname, filename), flatten = False)
                return numpy.transpose(img, (1, 0, 2))

def image_save(img, filename, dirname):
	''' _image_save(img, filename, img)

	Save image to file 
	filename: _string_ contains image filename
	img: 2D_numpy_array contains target image '''
	filename = os.path.join(dirname, filename)
	imsave(filename, img.transpose())
	return

def img_resize(img, r):
	''' img_resize(img, r)

	resize image for different scales
	img: numpy array of image
	r: image resize ratio, float
	Return: numpy array of image after resize '''
	return imresize(img, r, interp = 'bilinear')

def image_norm(img):
	'''_image_norm(img)

	Normalize the intensity of image, return intensity range 0-255
	img: 2D_numpy_array
	return 2D_numpy_array'''

	img = numpy.float32(img)
	img = img - numpy.min(img)
	if numpy.max(img) == 0:
		return numpy.uint8(img*255)
	else:
		return numpy.uint8(img / numpy.max(img) *255)

def image2edge(img, mode = None):
	'''_image2edge(img)

	convert image to edge map
	img: 2D_numpy_array 
	Return 2D_numpy_array '''
        if mode == 'canny':
                img = image_norm(img)
                edgeim = numpy.uint8(canny(img))*255
                return edgeim 
        if mode == 'sobel':
                img = image_norm(img)
                edgeim = sobel(img)*255
                return edgeim 
	img = numpy.float32(img)
	im1 = scipy.ndimage.filters.sobel(img,axis=0,mode='constant',cval =0.0)
	im2 = scipy.ndimage.filters.sobel(img,axis=1,mode='constant',cval =0.0)
	return (abs(im1) + abs(im2))/2

def unsharp(img):
        ''' unsharp(img)
        
        apply unsharp mask to the image
        img: original image array
        Return: image after unsharp masking, array like'''
        def unsharp2d(img):
                if len(img.shape) == 2:
                        blur = gaussian_filter(img, 50) 
                        blur = -0.1*blur
                        return blur + img
                else:
                        raise Exception('The image size is not recognized.')
        if len(img.shape) == 3 and img.shape[2] == 3:
                img[:, :, 0] = unsharp2d(img[:, :, 0])
                img[:, :, 1] = unsharp2d(img[:, :, 1])
                img[:, :, 2] = unsharp2d(img[:, :, 2])
        elif len(img.shape) == 2:
                img = unsharp2d(img)
        else:
                raise Exception('The image size  is not recognized.')
        return img

def imsize_reduce((m, n), *arg):
	''' imsize_reduce((m, n), img, coor)

	reduce the size of large image for faster processing and less memory use
	img: numpy array, original image
	coor: numpy array contains bounding box coordinates, it will be change size along with image
	(m, n): the maximum width and height of image, image larger than that size will be resized
	Return: image with new size '''

        if len(arg) == 1:
                img = arg[0]
                m0, n0 = img.shape
                if m0 >= n0 and m0 > m: # case: portait image
                        r = float(m)/float(m0)
                elif n0 >= m0 and n0 > n: # case: landscape image
                        r = float(n)/float(n0)
                else:
                        return img
                img = img_resize(img, r)
                return img

        elif len(arg) == 2:
                img = arg[0]
                coor = arg[1]
                m0, n0 = img.shape
                if m0 >= n0 and m0 > m: # case: portait image
                        r = float(m)/float(m0)
                elif n0 >= m0 and n0 > n: # case: landscape image
                        r = float(n)/float(n0)
                else:
                        return img, coor
                img = img_resize(img, r)
                coor = coor_resize(coor, r)		
                return img, coor
        else: 
                raise Exception('Unrecognized input argument!')

def imsize_normalize((m, n), *arg):
	''' imsize_normalize((m, n), img, coor)

	normalize the size of large image 
        img: numpy array, original image
	coor: numpy array contains bounding box coordinates, it will be change size along with image
	(m, n): the maximum width and height of image, image larger than that size will be resized
	Return: image with new size '''

        if len(arg) == 1:
                img = arg[0]
                m0, n0 = img.shape
                if m0 > n0 and m0 != m: # case: portait image
                        r = float(m)/float(m0)
                elif n0 > m0 and n0 != n: # case: landscape image
                        r = float(n)/float(n0)
                else:
                        return img
                img = img_resize(img, r)
                return img

        elif len(arg) == 2:
                img = arg[0]
                coor = arg[1]
                m0, n0 = img.shape
                if m0 > n0 and m0 != m: # case: portait image
                        r = float(m)/float(m0)
                elif n0 > m0 and n0 != n: # case: landscape image
                        r = float(n)/float(n0)
                else:
                        return img, coor
                img = img_resize(img, r)
                coor = coor_resize(coor, r)		
                return img, coor
        else: 
                raise Exception('Unrecognized input argument!')

def image2bb(img, thre = 128):
	''' _image2bb(img, thre)

	convert image hot map into bb text strings
	image: prediction image hotmap
	thre: threshold to binarize the hotmap
	Return: string, contains boundingbox coordinates'''
	lbimg = skimage_label(numpy.uint8(img > thre), background = 0)
	bbox = []
	for l in xrange(lbimg.max()+1):
		x, y = numpy.where(lbimg == l)
		x0 = min(x)
		y0 = min(y)
		x1 = max(x)
		y1 = max(y)
		bb = [x0, y0, x1, y1]
                bbox.append(bb)
	return bbox

def imageCrop(img, bb):
        '''imageCrop(img, bb)

        crop the image according to the bounding box bb
        bb: list of coordinates, x0, y0, x1, y1
        return: array, cropped image '''
        m = img.shape[0]
        n = img.shape[1]
        x0, y0, x1, y1 = bb
        x0 = max(0, x0)
        x1 = min(m, x1)
        y0 = max(0, y0)
        y1 = min(n, y1)
        p = img[x0:x1, y0:y1]
        return p

def image2roi(img, bb):
        '''image2roi(img, bb)

        convert images to stacks, which contains small regions of interest for each word
        input: 
                img, orginal image, array like
                bb, list of list, coordinates of bounding box
        output:
                list of array like objects contains region of interest'''

        s = []
        for b in bb:
                roi = imageCrop(img, b)
                s.append(roi)
        return s

def roi2image(roi, bb, imshape):
        '''roi2image(roi, bb, imshape)

        convert roi back to image
        input: 
                roi:image patch 
                bb: list of coordinates
                imshape: tuple of the image shape
        '''
        img = numpy.zeros(imshape)
        m = img.shape[0]
        n = img.shape[1]
        for b, r in zip(bb, roi):
                x0, y0, x1, y1 = b
                x0 = max(0, x0)
                x1 = min(m, x1)
                y0 = max(0, y0)
                y1 = min(n, y1)
                img[x0:x1, y0:y1] = r
        return img 


def boundPadding(p, m):
        ''' boundPadding(p)

        add lateral inhibitory around the image patch
        p: original image patch
        Return: image after zero padding. 
        '''
        m0 = p.shape[0]
        newp = numpy.zeros((m0+2*m, m0+2*m))
        newp[m:m+m0, m:m+m0] = p
        newp = imresize(newp, (m0, m0), 'bilinear')
        return newp

def drawbb(img, coor, color):
	'''  drawbb(canvas, coor, color)

	draw bounding box on canvas
	cavas: grey-scale image to draw on
	coor: coordinates of bounding box (tuple contains: x, y, w, h)
	color: the bounding box color '''

	def rect_draw(canvas, Coor):
		canvas = numpy.uint8(canvas)
		for coor in Coor:
			x1 = coor[0]
			y1 = coor[1]
			x2 = coor[2]
			y2 = coor[3]
			line1 = skimage_line(x1, y1, x2, y1)
			line2 = skimage_line(x1, y1, x1, y2)
			line3 = skimage_line(x1, y2, x2, y2)
			line4 = skimage_line(x2, y1, x2, y2)
			skimage_set_color(canvas, line1, 255)
			skimage_set_color(canvas, line2, 255)
			skimage_set_color(canvas, line3, 255)
			skimage_set_color(canvas, line4, 255)
		return canvas
		
	if len(img.shape) == 3 and img.shape[2] == 3: # if color image
		if color == 'r':
			img[:, :, 0] = rect_draw(img[:, :, 0], coor)
		elif color == 'g':
			img[:, :, 1] = rect_draw(img[:, :, 1], coor)
		elif color == 'b':
			img[:, :, 2] = rect_draw(img[:, :, 2], coor)
		else:
			print 'Error: use only \'r, g, b\' for colors'
			return

	else: # if greyscale image
		canvas = numpy.uint8(img)
		img = numpy.zeros((canvas.shape[0], canvas.shape[1], 3))
		img[:, :, 0] = canvas
		img[:, :, 1] = canvas
		img[:, :, 2] = canvas
		canvas = rect_draw(canvas, coor)
		if color == 'r':
			img[:, :, 0] = canvas
		elif color == 'g':
			img[:, :, 1] = canvas
		elif color == 'b':
			img[:, :, 2] = canvas
		else:
			print 'Error: use only \'r, g, b\' for colors'
			return
	return img 

def image2mser(greyim):
        ''' image2mser(greyim)

        convert greyscale image into mser regions
        input: 
                greyim: number array, grey scale image
        Return:
                numpy array contains label image'''
        mserimg = numpy.zeros(greyim.shape)
        mser = cv2.MSER()
        mser_areas = mser.detect(greyim, None)
        for i, m in enumerate(mser_areas):
                mserimg[m[:, 1], m[:, 0]] = i
        return mserimg

def imShow(img):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.imshow(img, interpolation = 'nearest')
        plt.axis('off')
        plt.show()
        return

def imreduce(M, img, coor = None):
        m, n = img.shape
        if coor:
                if m > M:
                        img = imresize(img, float(M)/float(m), 'bilinear')
                        coor = [int(float(i)*float(M)/float(m)) for i in coor]
                return img, coor
        else:
                if m > M:
                        img = imresize(img, float(M)/float(m), 'bilinear')
                return img

def pnt2cc(img, pnt):
        ccimg = numpy.zeros_like(pnt)
        sids = zip(*numpy.nonzero(pnt))  
        for i, coor in enumerate(sids):
                img0, coor0 = imreduce(32, img, coor)
                label = regiongrow(img0, coor0[0], coor0[1]) * i
                label = imresize(label, img.shape)
                ccimg = ccimg + label
        return ccimg

def regiongrow(crop, xs, ys):
    
        crop = numpy.int32(crop)
        # compute edge intensity
        eimg = sobel(crop)
        # define kernel for different orientation of edges
        k1 = numpy.asarray([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        k2 = numpy.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        k3 = numpy.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        k4 = numpy.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # edge at different orientations
        c1 = conv2(eimg, k1, 'same')
        c2 = conv2(eimg, k2, 'same')
        c3 = conv2(eimg, k3, 'same')
        c4 = conv2(eimg, k4, 'same')
        # combine them
        combimg = numpy.maximum(c1, numpy.maximum(c2, numpy.maximum(c3, c4)))

        # color difference in difference directions
        d3 = [[-1, -1], [1, 1]]
        dis3 = numpy.absolute(conv2(crop, d3, 'same'))
        d4 = [[1, 1], [-1, -1]]
        dis4 = numpy.absolute(conv2(crop, d4, 'same'))
        d1 = [[-1, 1], [-1, 1]]
        dis1 = numpy.absolute(conv2(crop, d1, 'same'))
        d2 = [[1, -1], [1, -1]]
        dis2 = numpy.absolute(conv2(crop, d2, 'same'))

        # initialize label image
        label = numpy.zeros_like(crop)
        label[0, :] = 1
        label[-1, :] = 1
        label[:, 0] = 1
        label[:, -1] = 1
        label[xs, ys] = 2

        # dilation kernel for different directions    
        selem1 = [[0, 1]]
        selem2 = [[1, 1, 0]]
        selem3 = [[0], [1]]
        selem4 = [[1], [1], [0]]
        maxe = numpy.percentile(numpy.abs(eimg), 80)

        i = 0
        while 1:
                i += 1 
                # dilation image: new regions
                dulllabel = numpy.uint8(label>0)
                nimg1 = dilation(dulllabel, selem1)  
                nimg2 = dilation(dulllabel, selem2) 
                nimg3 = dilation(dulllabel, selem3) 
                nimg4 = dilation(dulllabel, selem4) 

                # color differences at new regions
                diffimg1 = numpy.multiply(dis1, nimg1)
                diffimg2 = numpy.multiply(dis2, nimg2)
                diffimg3 = numpy.multiply(dis3, nimg3)
                diffimg4 = numpy.multiply(dis4, nimg4)

                # total dilation image
                nimg = dilation(dulllabel) - dulllabel 
                # cost function
                tcost = eimg + diffimg1 + diffimg2 + diffimg3 + diffimg4 + (nimg == 0)*maxint

                gp = numpy.argmin(tcost)
                xp, yp = numpy.unravel_index(gp, tcost.shape)

                # grows
                seedlabel = max([label[xp-1, yp], label[xp+1, yp], label[xp, yp-1], label[xp, yp+1]])
                label[xp, yp] = seedlabel
                if not i%100:
                        if not numpy.sum(label == 2):
                                return numpy.zeros_like(label)                                

                        if numpy.sum(numpy.multiply(numpy.uint8(label==2), numpy.abs(eimg)))/numpy.sum(label==2) > maxe :
                                return label == 2
                        if numpy.sum(label == 0) == 0:
                                if numpy.sum(numpy.multiply(numpy.uint8(label==2), numpy.abs(eimg)))/numpy.sum(label==2) > maxe * 0.5:
                                        return label == 2
                                else:
                                        return numpy.zeros_like(label)                                



