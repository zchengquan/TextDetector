#!/usr/bin/env python
#title           :main.py
#description     :The main program.
#author          :siyu zhu
#date            :Dec 16th, 2013
#version         :0.1
#usage           :python main.py
#notes           :
#python_version  :2.7

# import modules
import os
import numpy
import scipy
import pickle
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import zero_one_loss
from sklearn import ensemble
class AdaBoostClassifier:
	def __init__(self, no_of_stages = 20):
		self.no_of_stages = no_of_stages

	def fit(self, data, target):

		no_of_stages = self.no_of_stages	
		decision_stump = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1, max_features=1)
		#No. of samples
		m = data.shape[0]
		weight = numpy.ones(m)
		weight = numpy.float32(weight)/m

		Alpha = numpy.zeros(no_of_stages)
		classifiers = []
		for i in range(no_of_stages):
			decision_stump = decision_stump.fit(data, target, sample_weight = weight)
			classifiers.append(decision_stump)
			pred = decision_stump.predict(data)
			error = zero_one_loss(target, pred, normalize=True, sample_weight = weight)

			if error > 0.5:
				print 'error value is greater than 0.5!'

			beta = error/(1-error)
			if beta != 0: 
				weight[pred == target] = weight[pred==target]*beta
				weight = weight / weight.sum()
			print weight
			# beta_mat = (pred==target)*beta
			# beta_mat[beta_mat==0] = 1
			# weight = numpy.multiply(weight, beta_mat)
			if beta > 0:
				alpha = math.log(1/beta) 
			else:
				alpha = 10000 # make alpha extremly large if decision stump is totally correct.
			Alpha[i] = alpha
		self.Alpha = Alpha
		self.classifiers = classifiers

	def predict(self, data):
		no_of_stages = self.no_of_stages
		classifiers = self.classifiers
		Alpha = self.Alpha

		m = data.shape[0]
		Pred = numpy.zeros(m)
		for i in range(no_of_stages):
			decision_stump = classifiers[i]
			pred = decision_stump.predict(data)
			Pred += pred*Alpha[i]

		return Pred

if __name__ == '__main__':
	sample = load_iris()
	data = sample.data
	target = sample.target
	data1 = data[target == 0, :]
	target1 = target[target == 0]
	target1 = target1 - 1
	data2 = data[target == 1, :]
	target2 = target[target == 1]

	data = numpy.concatenate((data1, data2), axis = 0)
	target = numpy.concatenate((target1, target2), axis = 0)
	data_train, data_test, target_train, target_test = train_test_split(data, target)

	clf = AdaBoostClassifier(no_of_stages = 20)
	clf.fit(data_train, target_train)
	pred = clf.predict(data_test)
	pred = pred > 0

	clf1 = ensemble.AdaBoostClassifier(n_estimators = 20, algorithm = 'SAMME')
	clf1.fit(data_train, target_train)
	pred1 = clf1.predict(data_test)
	pred1 = pred > 0






