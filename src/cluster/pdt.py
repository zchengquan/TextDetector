#!/usr/bin/env python
#title           :pdt.py
#description     :perform pairwise dependency test.
#author          :siyu zhu
#date            :Jan 8th, 2013
#version         :0.1
#usage           :pdt()
#notes           :
#python_version  :2.7

#==============================================================================
# import modules
import numpy
from scipy import stats 

#==============================================================================

class pdt():
	
	def __init__(self, codebook, r):
		self.codebook = codebook
		self.r = r
		self.l = len(codebook)
		self.m = len(codebook[0, :])
		self.seed = self.codebook[0, :]
#		print self.r
#		print self.l
#		print self.m

	def dependency(self, x, y):
#		num = numpy.sum((x**2-numpy.mean(x**2, axis = 0))*(y**2-numpy.mean(y**2, axis = 0))) 
#		den = numpy.sqrt(numpy.sum((x**2-numpy.mean(x**2, axis = 0))**2)*numpy.sum((y**2-numpy.mean(y**2, axis = 0))**2))
#		d = num/den
		d = stats.pearsonr(x, y)
		return d[0]

	def list_r(self, z1):
		dep = []
		for i in range(0, self.l):
			z2 = self.codebook[i, :]
			d = -numpy.absolute(self.dependency(z1, z2))
			dep.append(d)
		indlist = numpy.argsort(dep)

		indlist = indlist[0:self.r]
		rcode = self.codebook[indlist, :]

		self.codebook = self.codebook[self.r:]
		self.l = len(self.codebook)
		if self.l > 0:
			self.seed = self.codebook[0, :]
		return rcode
	
	def exe(self):
		rcode = self.list_r(self.seed)
		return rcode		
