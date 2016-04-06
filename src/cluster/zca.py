#!/usr/bin/env python
#title           :zca.py
#description     :perform zca for feature space.
#author          :siyu zhu
#date            :Jan 6th, 2013
#version         :0.1
#usage           :zca()
#notes           :
#python_version  :2.7

#==============================================================================
# import modules

import numpy as np
from scipy import linalg
 
class ZCA():
 
	def __init__(self, regularization=10**-5):
		self.regularization = regularization
		 
	def whiten(self, X):
		self.mean_ = np.mean(X, axis=0)
		X -= self.mean_
		sigma = np.dot(X.T,X) / X.shape[1]
		U, S, V = linalg.svd(sigma)
		tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
		self.components_ = np.dot(tmp, U.T)
		X_transformed = np.dot(X, self.components_.T)
		return X_transformed
