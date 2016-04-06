#!/usr/bin/env python
#title           :plot.py
#description     :Test classifier and plot performance evaluation.
#author          :siyu zhu
#date            :March 17, 2014
#version         :0.1
#usage           :python plot.py
#notes           :
#python_version  :2.7

import matplotlib
matplotlib.use('Agg')
# import modules
import pickle
import os
import time
import numpy
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing


# define threshold Scaling 
def threScale(t):
    ''' make threshold plotable by making its length
	 equal to precision and recall vector '''
    t = numpy.append(t, t[-1])
    #t = t - numpy.min(t)
    #t = t / numpy.max(t)
    return t

# define F metric 
def fmetric(pre, rec):
    ''' compute f1 metric from precision and recall vector '''
    pre = pre
    rec = rec
    f = 2*pre*rec/(pre+rec+1e-10)
    return f
  
def error_curve(test_score, train_score, dirname):
    '''training error in each iteration,
	 (AdaBoost only)
    
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    p1 = plt.plot(test_score, label = 'test')
    p2 = plt.plot(train_score, label = 'train')
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(r'Testing curve')
    ax.legend(loc = 1)
    plt.savefig(os.path.join(dirname, 'error.png'))
    return
    
def roc_curve(fpr_te, tpr_te, dirname):
    ''' ROC curve

    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    p1 = plt.plot(fpr_te, tpr_te, label = 'roc')
    plt.grid(True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(r'ROC')
    ax.legend(loc = 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.savefig(os.path.join(dirname, 'roc.png'))
    return

def prf_curve(pre_te, rec_te, thre_te, dirname):
    ''' compute the precision, recall and f measure
    
    '''
    f1 = fmetric(pre_te, rec_te)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(thre_te, pre_te, label = 'precision')
    plt.plot(thre_te, rec_te, label = 'recall')
    plt.plot(thre_te, f1, label = 'f metric')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(r'Recall, Precision and F1-score')
    ax.legend(loc = 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    print 'Maximum F1:', max(f1)
    print 'at Preicision:', pre_te[numpy.argmax(f1)]
    print 'at Recall:', rec_te[numpy.argmax(f1)]
    print 'at Threshold:', thre_te[numpy.argmax(f1)]
    print 'Threshold minimum:', thre_te.min()
    print 'Threshold maximum:', thre_te.max()
    plt.savefig(os.path.join(dirname, 'prf.png'))
    return

def pr_curve(pre_te, rec_te, dirname):
    ''' Precision/Recall curve
    
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    p1 = plt.plot(pre_te, rec_te, label = 'p/r')
    plt.grid(True)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(r'Precision/Recall')
    ax.legend(loc = 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.savefig(os.path.join(dirname, 'pr.png'))
    return

def confMat(conf_arr, dirname):
	''' confusion matrix 

	conf_arr: n by n numpy array '''
	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
		tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)
	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(numpy.array(norm_conf), cmap=plt.cm.jet, 
			interpolation='nearest')
	width = len(conf_arr)
	height = len(conf_arr[0])

	for x in xrange(width):
	    for y in xrange(height):
		ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
			    horizontalalignment='center',
			    verticalalignment='center')
	cb = fig.colorbar(res)
	# alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	alphabet = ('Foreground', 'Background')
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.savefig(os.path.join(dirname, 'confusionMatrix.png'), format='png')

def sample_distribut(arr, dirname):
	'''plot samples distribute in differnet scales'''

	fig, ax = plt.subplots(figsize = (len(arr)/2, 5))
	ind = numpy.arange(len(arr))*4
	p1 = ax.bar(ind, arr, color = 'b')
	plt.grid(True)
	plt.xlabel('scale')
	plt.ylabel('')
	# ax.legend(p1[0], 'ICDAR2013')
	width = 0.5
	ax.set_xticks(ind+width)
	ax.set_xticklabels(range(len(arr)))

	def autolabel(rects):
		# attach some text labels
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
				ha='center', va='bottom')
	#autolabel(p1)
	plt.savefig(os.path.join(dirname, 'samDistr.png'))

def pplot(pred, lb, dirname):
	''' performance plot for all kinds of classifier that produce real value prediction 
	
	ROC curve plot,
	precision/recall curve plot,
	pre/rec/F-score curve plot '''
	fpr, tpr, t = metrics.roc_curve(lb, pred, pos_label = 1)
	pre, rec, thre = metrics.precision_recall_curve(lb, pred, pos_label=1)
	thre = threScale(thre)
	roc_curve(fpr, tpr, dirname)
	prf_curve(pre, rec, thre, dirname)
	pr_curve(pre, rec, dirname)
	return

def eplot(tescore, trscore, dirname):
	''' training error plot for adaboost classifier
	'''
	error_curve(tescore, trscore, dirname)
	return

def barplot(arr, dirname):
	''' bar plot for sample distribution 
	usually for across different scales '''
	sample_distribut(arr, dirname)
	


