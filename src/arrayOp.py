#!/usr/bin/env python
#title           :arrayOp.py
#description     :patch array and label array operation and transformations.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from arrayOp import * 
#notes           :
#python_version  :2.7

# import modules

import numpy
from scipy.misc import imresize
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
#------------------------------sample selection and permutation-----------------------------------------#
def rand_select(P, L):
	'''_rand_select( arr, n)

	Randomly select n samples from the array
	arr: _numpy_array_ to select from
	n: integer, the number of samples to select
	Return a smaller _numpy_array_ whose length is n ''' 
	ind = numpy.random.choice(len(L), len(L), replace = False)
	P = P[ind, ...]
	L = L[ind, ...]
	return P, L 

def SMOTE(T, N, k):
        """
        SMOTE(T, N, k)

        Populate samples with SMOTE algorithm
        Input:
                T: array-like, shape = [n_minority_samples, n_features]
                Holds the minority samples
                N: percetange of new synthetic samples: 
                n_synthetic_samples = N/100 * n_minority_samples.
                k: int. Number of nearest neighbours. 

        Returns
                array, shape = [(N/100) * n_minority_samples, n_features]"""
    

        n_minority_samples, n_features = T.shape

        if N < 100:
                N = 100
                pass

        if (N % 100) != 0:
                raise ValueError("N must be < 100 or multiple of 100")
    
        N = N/100
        n_synthetic_samples = N * n_minority_samples
        S = numpy.zeros(shape=(n_synthetic_samples, n_features))

        #Learn nearest neighbours
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(T)

        #Calculate synthetic samples
        for i in xrange(n_minority_samples):
                nn = neigh.kneighbors(T[i], return_distance=False)
                for n in xrange(N):
                        nn_index = choice(nn[0])
                        #NOTE: nn includes T[i], we don't want to select it 
                        while nn_index == i:
                                nn_index = choice(nn[0])
                
                        dif = T[nn_index] - T[i]
                        gap = numpy.random.random()
                        S[n + i * N, :] = T[i,:] + gap * dif[:]
        return S 

def equalization(P, L):
	''' _equalization( P, L)

	equalize foreground and background samples
	trim the longer vector to have the same length of the shorter
	P: sample features _numpy_array_ with shape m by n
	L: sample labels _numpy_array_ with shape m by 1 
	Return samples and features where foreground and background numbers
		are equal '''
	L = numpy.uint8(L) 
        P, L = rand_select(P, L)
	ind1 = L == 1 # 1 for foreground
	ind2 = L == 0 # 0 for background
	P1 = P[ind1, ...]
	L1 = L[ind1, ...]
	P2 = P[ind2, ...]
	L2 = L[ind2, ...]
	if len(L2) > len(L1):
		P2 = P2[:len(L1), ...]
		L2 = L2[:len(L1), ...]
	else:
		P1 = P1[:len(L2), ...]
		L1 = L1[:len(L2), ...]
	P = numpy.concatenate((P1, P2))
	L = numpy.concatenate((L1, L2))
	return P, L
#--------------------------------------------data reshape----------------------------------------------#
def flatten_patch(patch):
	''' _flatten_patch( patch)

	flatten patch matrix for further processing
	patch: 4D_numpy_array with shape (w, h, psize, psize)
	return: 2D_numpy_array with shape (w*h, psize*psize)'''
	return patch.reshape((
		patch.shape[0]*patch.shape[1],
		patch.shape[2]*patch.shape[3]*patch.shape[4]))

def flatten_label(label):
	'''_flatten_label( label)

	flatten label matrix
	label: 2D_numpy_array with shape (w, h)
	Return: 2D_numpy_array with shape (w*h,)'''
	return label.flatten()

def stack2array(patchStack, labelStack):
        if not len(patchStack) == len(labelStack):
            raise Exception('The label and patch stacks are not match!')
        flag = 0
        for i in range(len(patchStack)):
                p = patchStack[i]
                l = labelStack[i]
                p = flatten_patch(p)
                l = flatten_label(l)
                p, l = equalization(p, l)
                if sum(l) == 0:
                    continue
                if flag == 0:
                    patchArray = p
                    labelArray = l
                    flag = 1 
                else:
                    patchArray = numpy.concatenate((patchArray, p))
                    labelArray = numpy.concatenate((labelArray, l))
        if flag == 1:
                return patchArray, labelArray
        else:
                return  

def stack2image(arrstack, image = None):
        '''stack2image(arrstack)    

        convert different scales into one image_load
        arrstack: stack contains image from different scales
        Return: image with maximal pooling '''
        if not arrstack:
                return
        maxm = 0
        maxn = 0
        for arr in arrstack:
                m, n = arr.shape
                if m > maxm:
                        maxm = m
                        maxn = n
        img = numpy.zeros((maxm, maxn))
        for arr in arrstack:
                arrimg = imresize(arr, (maxm, maxn), interp = 'nearest')
                img = numpy.maximum(img, arrimg)
        try:
                img = imresize(numpy.transpose(image), img.shape)
                return img
        except:
                return img

def label2coor(label, thre):
        '''stack2mage(arrstack)    

        convert different scales into one image_load
        arrstack: stack contains image from different scales
        Return: image with maximal pooling '''
        if not label:
                return
        maxm = 0
        maxn = 0
        for arr in label:
                m, n = arr.shape
                if m > maxm:
                        maxm = m
                        maxn = n
        xs = []
        ys = []
        for arr in label:
                m, n = arr.shape
                arr = numpy.uint8(arr > thre)
                x, y = numpy.nonzero(arr)
                x = (x+0.5)/m * maxm  
                y = (y+0.5)/n * maxn 
                xs.append(x)
                ys.append(y)
        return xs, ys
#-----------------------------------------data range normarlization---------------------------------------#

def pred_norm(pred):
	''' _pred_norm(self, pred)

	Normalize prediction value, prediction value from classifiers
	range converted from -2 ~ 2 to 0 ~ 1 
	pred: 1D_numpy_array
	Return 1D_numpy_array '''
        try:
                pred = numpy.float32(pred)
                pred = pred - numpy.min(pred) 
                if numpy.max(pred) > 0:
                        return pred / numpy.max(pred)
                else:
                        return pred
        except:
                return pred 








