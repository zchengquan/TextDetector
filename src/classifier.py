#!/usr/bin/env python
#title           :detector.py
#description     :Generate training image patches in different scales.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :python detector.py dataset_dir classifier_type
#notes           :
#python_version  :2.7

# import modules

import numpy

from mail import mail
import plot
from arrayOp import *
from bbOp import *
from fileOp import *
from imgOp import *
from timeLog import timeLog

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.cross_validation import train_test_split

# No. of Jobs for parallel computing
MAIL_REC = 'junesiyu@gmail.com'

class classifier:

        def __init__(self, mode = 'adaboost'):
                
		if mode == 'adaboost':
			clf = GradientBoostingRegressor(
				learning_rate = 1,
				n_estimators = 1000,
				max_depth = 3,
				random_state = 0)
                elif mode == 'randomforest':
                        clf = RandomForestRegressor(
                                n_estimators = 10,
                                max_depth = None,
                                n_jobs = -1)  
		elif mode == 'SVM':
			clf = SVC(C = 10.0, 
				kernel = 'linear',
				)
		elif mode == 'vjcascade':
			clf = vjcascade(n_stage=30,
				n_esti = 1,
				l_rate = 1)
		elif mode == 'gridSearch':
			param_grid = [
			{'max_depth': [1, 2, 3], 'loss': ['ls', 'lad']},
			]
			gbr = GradientBoostingRegressor()
			clf = grid_search.GridSearchCV(gbr, param_grid, n_jobs = -1)
		else:
			raise Exception('no mode named: '+mode+' found!')

                self.classifier = clf
                self.mode = mode
                
        def data_load(self, datadir):
                mytime = timeLog('../timelogs/data_load')
                mytime.start()
		print 'Loading data ....',
		P, L = data_load(datadir)
		P = numpy.uint8(P)
		L = numpy.uint8(L)
                P_train, P_test, L_train, L_test = train_test_split(P, L, train_size = 0.8, test_size = 0.2, random_state = 22)
                self.feature = P_train
                self.label = L_train
                self.feature_test = P_test
                self.label_test = L_test
                mytime.end()
                mytime.final()

	def clf_train(self):

		'''  clf_train(self, clfname, mode = 'adaboost')

		train the classifier with feature generated
		clfname: _string_ contains filename of the classifier to save
		mode: {'adaboost', 'SVM', 'vjcascade', 'gridSearch'} _string_, 
			classifier type, where gridSearch is searched over 
			different types of SVM classifiers. '''

                mytime = timeLog('../timelogs/clf_train')
                mytime.start()

		print 'Training Classifier on ',
                print len(self.label), 'Samples'
		self.classifier = self.classifier.fit(self.feature, self.label)
		mail(MAIL_REC, '<Coding Message!!!>, clf_train_finished')
                mytime.end()
                mytime.final()
                return

	def clf_test(self):
		''' clf_test(self, clfname)

		test the classifier the features generated
		clfname:_string_ contains filename of the classifier 
		Return (pred, label) 1D_numpy_arrays for prediction and labels '''

                mytime = timeLog('../timelogs/clf_test')
                mytime.start()

		print 'Testing Classifier ...'
		plotdir = '../plot/'+time.ctime()
		make_dir(plotdir)
		# write classifier name to txt file
		open(os.path.join(plotdir, self.mode), 'a').close()
                if not 'decision_function' in dir(self.classifier):
                        self.classifier.decision_function = self.classifier.predict

		self.prediction = self.classifier.decision_function(self.feature_test)
                #plot.pplot(self.prediction, self.label_test, plotdir)

                #if self.mode == 'adaboost':
                #	test_score = []
                #        train_score = []
                #	for pred in self.classifier.staged_decision_function(self.feature_test):
                #		test_score.append(self.classifier.loss_(self.label_test, pred))
                #        for pred in self.classifier.staged_decision_function(self.feature):
                #                train_score.append(self.classifier.loss_(self.label, pred))
                #	plot.eplot(test_score, train_score, plotdir)
		# report 
		mail(MAIL_REC, '<Coding Message!!!>, clf_test_finished')
                mytime.end()
                mytime.final()
		return 

        def clf_load(self, clfname, clfdir):
                '''clf_load(clfname, clfdir)
                
                load classsifier from pickle
                clfname: classifier file name
                clfdir: directory contains the file
                Return: classifier structure '''
                print os.path.join(clfdir, clfname)
                self.classifier = pickle.load(open(os.path.join(clfdir, clfname)))
                return

        def clf_save(self, clfname, clfdir):
                '''clf_save(clf, clfname, clfdir)
                save classifier to pickle
                clf: classifier structure to save
                clfname: filename to save
                clfdir: directory contain classifier file'''
                make_dir(clfdir)
                pickle.dump(self.classifier, open(os.path.join(clfdir, clfname), 'wb'))
                return

