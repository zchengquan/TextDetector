#!/usr/bin/env python
#title           :textDet.py
#description     :test text detector.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :python textDet.py dataset_dir classifier_type
#notes           :
#python_version  :2.7


# import modules
import numpy
import os
from fileOp import codebook_load, pickle_save, pickle_load
from sys import argv
import getopt
from detector import detector
from classifier import classifier
from parseData import parseWord2003, parseWord2013, parseChar2013
from wordGraph import wordGraph_test, wordbb2pred
#---------------------------------test harness------------------------------------------#

def main():
    # define parameters ###########################################################:
    expname = 'myexpe/'
    data = 'icdar2013word' # data for training/testing
    eMode = True # edge detection
    CodeBookName1 = '../codebooks/Patch/codeBook.npy' # codebook name
    CodeBookName2 =   '../codebooks/Verify/codeBook.npy' # codebook name

    coarseclfname = 'coarse'
    fineclfname = 'fine'
    wordgraphclfname = 'wordgraph'
    pdirname = '../data/' # dir contains all experiment data
    cdirname = os.path.join(pdirname, expname)
    clfdir = os.path.join(cdirname, 'clf/') # dir to save classifier
    rawdir = os.path.join(cdirname, 'raw/') # dir for original image
    npydir = os.path.join(cdirname, 'npy/') # dir for feature and label npy
    roitestdir = os.path.join(cdirname, 'roitest/') # dir for region of interest fine detector
    predir = os.path.join(cdirname, 'pre/') # dir for preprocessing
    predtxtdir = os.path.join(cdirname, 'pretxt/') # dir for txt file of bounding boxes.
    txtdir = os.path.join(cdirname, 'txt/') # dir for bounding box txt files
    # applying coarse detector ###########################################################:
    mode = 'adaboost' # classification mode for detector
    lMode = 'foreground' # foreground/whitespace
    fMode = 'context' # local or contextual
    psize = 32
    ssize = 16
    nob = 3
    ratio = 0.9
    rrange = 30
    para0 = (float(psize - ssize)/psize)**2
    para1 = 1 - ratio
    rpower = ratio ** numpy.asarray(range(rrange))

    data = pickle_load('detect', cdirname)
    codebook = codebook_load(CodeBookName1)
    myDetector = detector(codebook, data,
            psize, ssize,
            nob, rpower,
            para0, para1,
            lMode, fMode, eMode)
    myClassifier = classifier()
    myClassifier.clf_load(coarseclfname, clfdir)
    myDetector.image_test(rawdir, predir, myClassifier.classifier)
    # applying fine detector and region growing ###########################################################:
    mode = 'adaboost' # classification mode for detector
    lMode = 'foreground' # foreground/whitespace
    fMode = 'local' # local or contextual
    rpower = ratio ** numpy.asarray(range(rrange))
    codebook = codebook_load(CodeBookName2)
    data = pickle_load('region', cdirname)
    myDetector = detector(codebook, data,
            psize, ssize, nob, rpower,
            para0, para1,
            lMode, fMode, eMode)
    myClassifier = classifier(mode)
    myClassifier.clf_load(fineclfname, clfdir)
    myDetector.roi_test(predir, rawdir, roitestdir, myClassifier.classifier)
    # applying word graph ###########################################################:
    myClassifier = classifier()
    myClassifier.clf_load(wordgraphclfname, clfdir)
    wordbb = wordGraph_test(roitestdir, myClassifier.classifier)
    wordbb2pred(wordbb, predtxtdir)

if __name__ == '__main__':
    main()
