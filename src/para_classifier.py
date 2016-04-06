#!/usr/bin/env python
#title           :para_classifier.py
#description     :kernel functions for parallel computing used in Agglomerative training.
#author          :siyu zhu
#date            :Feb 3, 2015
#version         :0.1
#usage           :from para_classifier import para_adaboost
#notes           :
#python_version  :2.7

# import modules
import numpy
import scipy
from sklearn.ensemble import GradientBoostingRegressor

#--------------------BEGIN: functions used for parallel computation---------------------#
def para_adaboost(data):
	''' para_adaboost(data)

	kernel function for parallel computing adaboost classifier
	data: training data containing features and labels in a tuple 
	Return: adaboost classifier model '''
	model = GradientBoostingRegressor(
		learning_rate = 1,
		n_estimators = 1000,
		max_depth = 1,
		random_state = 0
		)	
	patch, label = data
	model = model.fit(patch, label)
	return model
#-----------------END: functions used for parallel computation--------------------------#


