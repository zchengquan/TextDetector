#!/usr/bin/env python
#title           :ckmean.py
#description     :perform convolutional K means on training image pathces
#author          :siyu zhu
#date            :Dec 16th, 2013
#version         :0.1
#usage           :
#notes           :
#python_version  :2.7
#==============================================================================
# import modules
import os
import numpy
import matplotlib
import math, random
import Image, ImageMath, ImageOps
import copy
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import linalg as linalg

#==============================================================================
class ckmean:
	# define class name 
	def colnorm(self,mat):
		# normalize columns of a matrix, that each column has a magnitude equals 1.
		# INPUT:
		# mat: matrix for processing
		# OUTPUT:
		# mat_new: each column has norm equals 1

		mat_new = mat
		for i in range(0, len(mat)):
			col = mat[i, :]
			# nomalize the vector
			norm_col = numpy.linalg.norm(col)
			if norm_col == 0:
				col = col
			else:
				col = col/norm_col
				mat_new[i, :] = col
		return(mat_new)

	def __init__(self, data, kk):
		# Convolutional K-means 
		# INPUT:
		# data: matrix each column is a sample vector
		# kk: number of total clusters
		# ii: number of iterations for kmeans training
		# OUTPUT:
		# D: matrix containing center vectors in columns"""

		print('starting kmeans quatization...(.py file is used)')
		# Initialization of D by randomly pick from training data
		col_idx = random.sample(range(0, len(data)), kk)
		D = data[col_idx, :]
		D = self.colnorm(D)
		self.data = data
		self.kk = kk
		self.D = D

	def ckmeans_update(self):
		# update center matrix D and hot encoding for convolutional k-means
		# INPUT:
		# D: initial estimate of center matrix
		# data: training sample matrix, each column is a sample
		# OUTPUT:
		# D: new cluter center matrix after one time update 
		
		D = self.D
		data = self.data
		kk = self.kk
		maxarg = numpy.empty((len(data),), dtype = int)
		maxidx = numpy.empty((len(data),), dtype = int)
		# loop across each sample data """
		for i in range(0, len(data)):
			# print ('len of data =', i)
			arg = numpy.empty((len(D),), dtype = int)
			x = data[i, :]
			# loop across each column of D """
			for j in range(0, len(D)):
				col = D[j, :]
				arg[j] = numpy.dot(col.T, x)
			maxarg[i] = numpy.max(arg[:])
			maxidx[i] = arg.argmax()
#		self.pro = numpy.sum(maxarg)
		# Update columns of D """
		for i in range(0, kk):
			# print ('len of D = ', i)
			idx = numpy.where(maxidx==i)
			data_i = data[idx, :]
			data_i = data_i[0, :, :]
			s_i = maxarg[idx]
			if len(s_i) != 0:
				multip = data_i.T*s_i
				summ_num = numpy.sum(multip, axis=1)
				summ_den = numpy.sum(s_i)
				D[i, :] = summ_num/summ_den
		# normalize each column after update """
		D = self.colnorm(D)
		self.D = D


