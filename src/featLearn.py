#!/usr/bin/env python
#title           :featLearn.py
#description     :utilize cython code ckmean.pyx to learn features
#author          :siyu zhu
#date            :Dec 16th, 2013
#version         :0.1
#usage           :
#notes           :
#python_version  :2.7
#==============================================================================
# import modules
import numpy
import scipy
import os
from detector import detector
from parseData import parseWord2003, parseWord2013, parseChar2013
from fileOp import data_load, pickle_save, codebook_load, pickle_load
from cluster import ckmean
from random import randint
#==============================================================================

def dataSample(data, psize = 8, repeatNum = 5):
        ''' dataSample(data, psize = 8, repeatNum = 5)

        randomly sample smaller patches from larger patches
        Input:
                data: array like, larger patch array, number of samples by number of features
                psize: smaller patch size
                repeatNum: number of small patches to generate from one large patch
        Output:
                array like, number of smaller samples by number of features (psize * psize)'''
        if len(data.shape) == 2:
                m, n = data.shape
                data = numpy.reshape(data, (m, int(n**0.5), int(n**0.5)))
        m, n, _ = data.shape
        sample = numpy.empty((m*repeatNum, psize, psize))
        k = 0
        for d in data:
                for i in range(repeatNum):
                        rind1 = randint(0, n-psize)
                        rind2 = randint(0, n-psize)
                        sample[k, ...] =  d[rind1:rind1+psize, rind2:rind2+psize]
                        k += 1
        return numpy.reshape(sample, (m*repeatNum, psize*psize))

def featLearn(data = 'icdar2013word'):

	# define parameters ###########################################################:
	psize = 32 # size of image patch
	ssize = 16 # step size
	ratio = 0.90 # ratio that each time image resized by
	rpower = ratio**numpy.asarray(range(30)) # number of power ratio is times by
	para0 = 0.8 # overlapping threshold from Coates paper
	para1 = 0.3 # width/height threshold from Coates paper
	nob = 3 # number of blocks
        expname = 'ft0/' # current experiment dir

	codeBookName = '../codebooks/codebook/codeBook.npy' # codebook name
	lMode = 'foreground' # {'foreground', 'whitespace', 'whitespace_strict'}
	fMode = 'local' #{'original' (no convolution), 'local', 'context'}
        eMode = False

        pdirname = '../data/' # dir contains all experiment data
        parsedGTName = 'detect'
        cdirname = os.path.join(pdirname, expname)
	labdir = os.path.join(cdirname, 'raw/') # dir for original image
	npydir = os.path.join(cdirname, 'npy/') # dir for feature and label npy

        # parse data ###################################################################:
	if data == 'icdar2003word':
		# define direcotries and filenames:
		imdir = '../icdar2003/icdar2003/SceneTrialTest' # containing original image
		xmlfilename = '../icdar2003/icdar2003/SceneTrialTest/locations.xml'
		myParser = parseWord2003()
		groundTruth = myParser.parseData(imdir, xmlfilename)

	elif data == 'icdar2013word':
		imdir = '../icdar2013/task21_22/train/image' # containing original image
		bbdir = '../icdar2013/task21_22/train/word_label' # containing bb text files.
                #imdir = '../icdar2013/task21_22/test/image' # containing original image
                #bbdir = '../icdar2013/task21_22/test/word_label' # containing bb text files.
		myParser = parseWord2013()
		groundTruth = myParser.parseData(imdir, bbdir)

	elif data == 'icdar2013char':
		imdir = '../icdar2013/task21_22/train/image' # containing original image
		bbdir = '../icdar2013/task21_22/train/char_label' # containing bb text files
		myParser = parseChar2013()
		groundTruth = myParser.parseData(imdir, bbdir)

	else:
		raise Exception('No data named:'+data+' found!')
        groundTruth = myParser.prepareImg(groundTruth, imdir, labdir)
        pickle_save(groundTruth, parsedGTName, cdirname)

        # extract features ############################################################:
        groundTruth = pickle_load(parsedGTName, cdirname)
        codebook = codebook_load(codeBookName)
        myDetector = detector(codebook, groundTruth,
                psize, ssize, nob, rpower,
                para0, para1,
                lMode, fMode, eMode)
        myDetector.image_train(labdir, npydir)

	# load image patches
        kk = 1000
        ii = 1000
        data = dataSample(data)
	w = numpy.ones(data.shape[0])
	D = ckmean.init(data, kk)
	D, idx, dm  = ckmean.update(data, D, ii, w)
        cbdir = '../codebooks/codebook/'
	numpy.save(os.path.join(cbdir, 'codeBook'), D)
	numpy.save(os.path.join(cbdir, 'codeBookIdx'), idx)
	numpy.save(os.path.join(cbdir, 'codeBookErr'), dm)
        #return D, idx, dm

if __name__ == '__main__':
	featLearn()
