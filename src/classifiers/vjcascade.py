#!/usr/bin/env python
#title           :vjcascade.py
#description     :Viola-jones cascade classifier.
#author          :siyu zhu
#date            :Sep 29, 2014
#version         :0.1
#notes           :
#python_version  :2.7

#==============================================================================
# import modules
import numpy
import pdb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class vjcascade:
	def __init__(self, n_stage, n_esti, l_rate):
		self.n_stage = n_stage # number of stages for cascade
		self.n_esti = n_esti # number of estimates
		self.l_rate = l_rate
		self.rec_thre = 0.99
		self.pre_thre = 0.51 

	def sample_prune(self):
                ''' sample_prune(self)

                check current stage performance, and go to next stage if qualify
                Input:
                        self,
                Return:
                        1, if current stage qualify and goes to next stage;
                        2, if current stage get perfect performance on training data;
                        0, if not qualify. '''
		# check the validation of current stage performance
		precision, recall, thresholds = precision_recall_curve(self.lb,
			 self.pred, pos_label=1)
		thresholds = numpy.append(thresholds, thresholds[-1]) # make <thresholds> same length as <recall>
		# pdb.set_trace()
		self.R = recall[recall >= self.rec_thre][-1]
		self.P = precision[recall >= self.rec_thre][-1]
		self.TH = thresholds[recall >= self.rec_thre][-1]
		if self.P >= self.pre_thre and self.P < 1:
			pred_label = numpy.uint8(self.pred >= self.TH)
			print 'Valid, go to next stage'
			print 'Threshold:', self.TH  
			print 'Pre:',precision_score(self.lb,pred_label,pos_label=1)
			print 'Rec:',recall_score(self.lb,pred_label,pos_label=1)
			self.lb = self.lb[pred_label == 1]
			self.ft = self.ft[pred_label == 1, :]
			return 1
		elif self.P == 1:
			return 2
		else:
			return 0

	def stage_train(self):
		# train stage classifier
		print 'Current Stage:', self.j
		print 'Samples Left:', len(self.ft)
		self.stageclf = GradientBoostingRegressor(learning_rate = self.l_rate,
			 n_estimators = self.n_esti,
			 random_state = None )
		self.stageclf = self.stageclf.fit(self.ft, self.lb)
		self.pred = self.stageclf.predict(self.ft)
		sp = self.sample_prune()
		if sp == 1:
			self.CLF.append(self.stageclf)
			self.Thred.append(self.TH)
			self.Precision.append(self.P)	
			self.Recall.append(self.R)
			self.j += 1	
		elif sp == 2:
			self.j = self.n_stage
			print 'Stop Training'
			print '100% Precision reached!'

		else:
			print 'Evaluation NOT valid'
			if self.j < 10:
				self.n_esti += 2
			else:
				self.n_esti += 20
			print "Stage Features Increased to", self.n_esti 
		return 

	def fit(self, ft, lb):
		# train using data ft, and label lb.
		self.ft = ft
		self.lb = lb
		self.j = 0 # number of stage iterator	
		self.CLF = []	
		self.Precision = []
		self.Recall = []
		self.Thred = []
		# Loop for stages
		while self.j < self.n_stage:	
			self.stage_train()
		print 'Viola-Jones Trained'
		self.n_stage = len(self.CLF) # No. of stages
		return self

	def predict(self, ft):
		# predict sample labels using trained classifier
		lb = numpy.ones(len(ft))
		for j in range(self.n_stage):
			# print 'active samples left:', sum(lb)
			pred = self.CLF[j].predict(ft)
			lb[pred <= self.Thred[j]] = 0
		return lb

