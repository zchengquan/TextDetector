#!/usr/bin/env python
#title           :featExt.py
#description     :kernel functions for parallel computing used in textDetect.py.
#author          :siyu zhu
#date            :Nov 5th, 2014
#version         :0.1
#usage           :from featExt import featExt 
#notes           :
#python_version  :2.7

# import modules
import numpy
import scipy
from scipy.misc import imresize
from scipy.signal import convolve2d
from imgOp import image_norm

#--------------------BEGIN: functions used for parallel computation---------------------#

def maximalResize(patch, nob = 3, st = 10, re = 2):
        '''maximalResize(patch, nob = 3, st = 10, re = 2)
        
        resize patch using maximal pooling
        nob: int, number of blocks,
        st: step size (psize/nob)
        re: residue value (psize%nob)
        Return array resized patch '''
        p = numpy.empty((nob, nob))
        for i in xrange(nob):
                for j in xrange(nob):
                    p[i, j] = numpy.max(patch[i*st:(i+1)*st+re, j*st:(j+1)*st+re])
        return p

def averageResize(patch, nob):
        '''averageResize(patch, nob)
        
        resize patch using average pooling
        patch: array of original patch
        nob: number of blocks
        Return array of patch '''
        return imresize(patch, (nob, nob), interp = 'nearest')

def conv2d(im, co):
	''' conv2d(im, co)

	convolve an image with a codebook patch 
	im: 2D_numpy_array contains target image, with shape m by n
	co: 2D_numpy_array codebook patch with shape 8 by 8 
	Return 2D_numpy_array with shape same as <im> '''
	return convolve2d(im, co, mode = 'same')
	
def get_patch(img, psize, ssize):
	'''  get_patch1(img, psize, ssize)

	generate patch features from image
	img: 2D_numpy_arrray_ contains image
	psize: _integer_ patch size
	ssize: _integer_ step size
	Return 4D_numpy_array_ with shape (w, h, psize, psize)'''

	(m, n) = map(lambda var:int(var), img.shape)
	(ind1, ind2) = map(lambda var:range(0, var-psize, ssize), (m, n))
	patch = numpy.empty((len(ind1), len(ind2), psize, psize))
        for p, i in enumerate(ind1):
                for q, j in enumerate(ind2):
                        patch[p, q, :, :] = img[i:i+psize, j:j+psize]
	return patch 
    
def get_patch1(img, psize, ssize, nob, mode = 'maximal'):
	'''  get_patch1(img, psize, ssize)

	generate patch features from image
	img: 2D_numpy_arrray_ contains image
	psize: _integer_ patch size
	ssize: _integer_ step size
	Return 4D_numpy_array_ with shape (w, h, psize, psize)'''

	(m, n) = map(lambda var:int(var), img.shape)
	(ind1, ind2) = map(lambda var:range(0, var-psize, ssize), (m, n))
	patch = numpy.empty((len(ind1), len(ind2), nob, nob))
        if mode == 'maximal':
                st = int(psize/nob)
                re = int(psize%nob)
                for p, i in enumerate(ind1):
                        for q, j in enumerate(ind2):
                                cpat = img[i:i+psize, j:j+psize]
                                patch[p, q, :, :] = maximalResize(cpat, nob, st, re)
        elif mode == 'normal':
                for p, i in enumerate(ind1):
                        for q, j in enumerate(ind2):
                                cpat = img[i:i+psize, j:j+psize]
                                patch[p, q, :, :] = averageResize(cpat, nob)
        else:
                raise Exception('mode is not found!')
	return patch 

def get_patch2(img, psize, ssize, nob, mode = 'maximal'):
	'''  get_patch2(img, psize, ssize, nob)

	generate patch features from image, include context information
	img: 2D_numpy_arrray_ contains image
	psize: _integer_ patch size
	ssize: _integer_ step size
	nob: number of blocks for local area
	Return 4D_numpy_array_ with shape (w, h, psize, psize)'''

	nob = nob*3 # the total number of blocks are 3 times than local patch
	(ind1, ind2) = map(lambda var:range(psize, var, ssize), img.shape)
	# expand image with zero padding
	img1 = numpy.zeros((img.shape[0]+2*psize, img.shape[1]+2*psize), dtype = numpy.uint8)
	img1[psize:img.shape[0]+psize, psize:img.shape[1]+psize] = img
	# extract patch with context
	patch = numpy.empty((len(ind1), len(ind2), nob, nob), dtype = numpy.uint8)
        if mode == 'maximal':
                st = int(psize*3/nob) # block size in terms of pixels
                re = int(psize*3%nob) # residue, overlapping pixels between blocks
                for p, i in enumerate(ind1):
                        for q, j in enumerate(ind2):
                                # patch with context
                                cpat = img1[i-psize:i+2*psize, j-psize:j+2*psize]
                                patch[p, q, :, :] = maximalResize(cpat, nob, st, re)
        elif mode == 'normal':
                for p, i in enumerate(ind1):
                        for q, j in enumerate(ind2):
                                cpat = img1[i-psize:i+2*psize, j-psize:j+2*psize]
                                patch[p, q, :, :] = averageResize(cpat, nob)
        else:
                raise Exception('mode is not found!')
	return patch 

def featExt(img, codeim, (psize, ssize), nob, mode = 'local'):
	''' para_featExt(img, codeim, (m0, n0), (psize, ssize), nob)

	kernel function used in parallel processing 
	This function performs:
			 convolution,
			 patch extraction,
			 spatial pooling
	img: 2D_numpy_array with shape m by n
	codeim: 2D_numpy_array codebook patch with shape 8 by 8
	(m0, n0): _tuple_ with _integer_, original image size before resize
	(psize, ssize): _tuple_ with _integer_, patch sizes, step sizes
	nob: _integer_, number of blocks (default = 3)
	Return 2D_numpy_array convolutional features with shape
		 No. of samples X No. of features '''

        cim = conv2d(img, codeim) # convolve image
        cim = image_norm(cim) # normalize intensity values
        #cim = imresize(cim, (m0, n0), interp = 'nearest') # resize image to original
	if mode == 'local':
		p = get_patch1(cim, psize, ssize, nob, mode = 'maximal') #extract patch based on current patch size
	elif mode == 'context':
		p = get_patch2(cim, psize, ssize, nob, mode = 'maximal') #extract patches with context 
	else:
		raise Exception('mode must be local or context')	
	return p # return patch features

#-----------------END: functions used for parallel computation--------------------------#


