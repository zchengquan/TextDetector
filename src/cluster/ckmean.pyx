#!/usr/bin/env python
#title           :ckmean.pyx
#description     :perform convolutional K means on training image pathces
#author          :siyu zhu
#date            :Jan 10, 2013
#version         :0.1
#usage           :
#notes           :
#python_version  :2.7
#cython_version  :0.19.2


# import modules
import os
import numpy
import copy
import time

# import cython compile module
cimport numpy as numpy
DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t	
ctypedef numpy.long_t DTYPEINT_t

cdef numpy.ndarray[DTYPE_t, ndim = 2] colnorm(numpy.ndarray[DTYPE_t, ndim = 2] mat):
	''' numpy.ndarray[DTYPE_t, ndim = 2] colnorm(numpy.ndarray[DTYPE_t, ndim = 2] mat)
	
	normalize columns of a matrix
	mat: matrix for processing
	Return: each column has vector length = 1 '''
	cdef DTYPE_t norm, a
	cdef Py_ssize_t i, j
	cdef m, n
	m = mat.shape[0]
	n = mat.shape[1]

	for i in xrange(m):
		norm = 0
		for j in xrange(n):
			a = mat[i, j]
			norm += a**2
		norm = norm **0.5
		if norm != 0:
			for j in range(n):
				mat[i, j] = mat[i, j] / norm
	return mat 

def init(numpy.ndarray[DTYPE_t, ndim = 2] data, int kk):
	''' init(numpy.ndarray[DTYPE_t, ndim = 2] data, int kk)

	Convolutional K-means 
	data: matrix each column is a sample vector
	kk: number of total clusters
	Return: matrix containing center vectors in columns '''

	print('starting kmeans quatization...(pyx file is used)')
	# Initialization of D by randomly pick from training data
	cdef numpy.ndarray[DTYPEINT_t, ndim = 1] col_idx = numpy.zeros(data.shape[1], dtype = int)
	cdef numpy.ndarray[DTYPE_t, ndim = 2]  D = numpy.zeros([kk, data.shape[1]], dtype = DTYPE)	

	# col_idx = random.sample(range(0, len(data)), kk)
	col_idx = numpy.random.choice(data.shape[0], kk, replace = False)
	D = data[col_idx, :]
	return colnorm(D)

def update(numpy.ndarray[DTYPE_t, ndim = 2] data,
	 numpy.ndarray[DTYPE_t, ndim = 2] D,
	 int ii,
	 numpy.ndarray[DTYPE_t, ndim = 1] weight):
	''' update(numpy.ndarray[DTYPE_t, ndim = 2] data, numpy.ndarray[DTYPE_t, ndim = 2] D, int ii, numpy.ndarray[DTYPE_t, ndim = 1] weight)
	update center matrix D and hot encoding for convolutional k-means

	D: initial estimate of center matrix
	data: training sample matrix, each column is a sample
	ii: number of iterations for kmeans training
	weight: numpy array contains sample weight
	Return: new cluter center matrix after one time update '''

	print "Start K-means learning with iterations = ", ii
	assert data.dtype == DTYPE and D.dtype == DTYPE
	cdef int m, n, k
	m = data.shape[0]
	n = D.shape[0]
	k = data.shape[1]
	cdef Py_ssize_t i, j, q, p
	cdef DTYPE_t value, idx, dotpro, sum_value, djq
	cdef numpy.ndarray[DTYPE_t, ndim = 1] sum_col = numpy.zeros(k, dtype = DTYPE)
	cdef numpy.ndarray[DTYPE_t, ndim = 1] maxarg = numpy.zeros(m, dtype = DTYPE)
	cdef numpy.ndarray[DTYPE_t, ndim = 1] maxidx = numpy.zeros(m, dtype = DTYPE)
	cdef numpy.ndarray[DTYPE_t, ndim = 1] dm = numpy.zeros(ii, dtype = DTYPE)
	
	# loop for all iterations
	for p in xrange(ii):
		print 'iteration', p
		D1 = copy.copy(D) # record center at the beginning of each loop
		# loop across each sample data """
		for i in xrange(m):
			value = -10000
			idx = 0
			# loop across each column of D """
			for j in xrange(n):
				dotpro = 0	
				for q in xrange(k):
					dotpro += D[j, q] * data[i, q]
				if dotpro > value:
					value = dotpro
					idx = j	

			maxarg[i] = value * weight[i]
			maxidx[i] = idx
		######################################################
		# Update columns of D """
		for i in xrange(n):
			sum_value = 0
			sum_col = numpy.zeros(k, dtype= DTYPE)
			# search for each data column	
			for j in xrange(m):
				if maxidx[j] == i:
					value = maxarg[j]
					sum_value += value
					
					for q in xrange(k):
						sum_col[q] += data[j, q] * value 
			if sum_value != 0:
				for q in xrange(k):
					D[i, q] = sum_col[q]/sum_value
		#######################################################
		# normalize each column after update """
		D = colnorm(D)
		# compute the center movement SSD
		D2 = copy.copy(D) # record center at the end of each loop
		djq = 0
		for j in xrange(n):
			for q in xrange(k):
				djq += (D1[j,q] - D2[j, q])**2
		dm[p] = djq**0.5
	return D, maxidx, dm	
		

if __name__ == '__main__':
	data = numpy.load('icdar2003word/P1.npy')
	kk = 1000	
	ii = 1000
	weight = numpy.ones(data.shape[0])
	D = init(data, kk)	
	D, idx, dm = update(data, D, ii, weight)
	numpy.save('codeBook', D)
	numpy.save('codeBookIdx', idx)
	numpy.save('codeBookErr', dm)
