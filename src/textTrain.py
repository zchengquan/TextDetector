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
#---------------------------------test harness------------------------------------------#

def main():
        expname = 'myexpe/'
        log = 'generate ground truth bounding box for fine detector test \r\n'  
        data = 'icdar2013word' # data for training/testing 
        eMode = True # edge detection
        coarseCodeBookName = '../codebooks/coarseDet/codeBook.npy' # codebook name
        fineCodeBookName =   '../codebooks/fineDet/codeBook.npy' # codebook name

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

        # define parameters ###########################################################:

        coarseclfname = 'coarseDet'
        fineclfname = 'fineDet3'
        pdirname = '../data/' # dir contains all experiment data
        datalist = 'datalist'
        cdirname = os.path.join(pdirname, expname)
        clfdir = os.path.join(cdirname, 'clf/') # dir to save classifier
        rawdir = os.path.join(cdirname, 'raw/') # dir for original image
        npydir = os.path.join(cdirname, 'npy/') # dir for feature and label npy
        roidir = os.path.join(cdirname, 'roi/') # dir for region of interest of coarse detector
        roitestdir = os.path.join(cdirname, 'roitest/') # dir for region of interest fine detector
        predir = os.path.join(cdirname, 'pre/') # dir for preprocessing
        preMapdir = os.path.join(cdirname, 'preMap/') # dir for preprocessing hotmaps
        predtxtdir = os.path.join(cdirname, 'pretxt/') # dir for txt file of bounding boxes.
        resdir = os.path.join(cdirname, 'res/') # dir for results
        mapdir = os.path.join(cdirname, 'map/') # dir for hotmaps
        pmapdir = os.path.join(cdirname, 'pmap/') # dir for pixel maps
        txtdir = os.path.join(cdirname, 'txt/') # dir for bounding box txt files
        # write log file, a simple discription of experiment
        with open(os.path.join(cdirname, 'log.txt'), 'a') as f:
                f.write(log)

        # parse data ###################################################################:
        if data == 'icdar2003word':
                # define direcotries and filenames:
                imdir = '../icdar2003/icdar2003/SceneTrialTest' # containing original image
                xmlfilename = '../icdar2003/icdar2003/SceneTrialTest/locations.xml'
                myParser = parseWord2003()
                dataList = myParser.parseData(imdir, xmlfilename)

        elif data == 'icdar2013word':
                #imdir = '../icdar2013/task21_22/train/image' # containing original image
                #bbdir = '../icdar2013/task21_22/train/word_label' # containing bb text files.
                imdir = '../icdar2013/task21_22/test/image' # containing original image
                bbdir = '../icdar2013/task21_22/test/word_label' # containing bb text files.
                myParser = parseWord2013()
                dataList = myParser.parseData(imdir, bbdir)

        elif data == 'icdar2013char':
                imdir = '../icdar2013/task21_22/train/image' # containing original image
                bbdir = '../icdar2013/task21_22/train/char_label' # containing bb text files
                myParser = parseChar2013()
                dataList = myParser.parseData(imdir, bbdir)

        else:
                raise Exception('No data named:'+data+' found!')

        dataList = myParser.prepareImg(dataList, imdir, rawdir)
        pickle_save(dataList, datalist, cdirname)
        # extract features ############################################################:
        dataList = pickle_load(datalist, cdirname)
        codebook = codebook_load(coarseCodeBookName)
        myDetector = detector(codebook, dataList,
                psize, ssize, nob, rpower,
                para0, para1,
                lMode, fMode, eMode )

        myDetector.image_train(rawdir, npydir)
        # training classsifier ########################################################:
        myClassifier = classifier(mode)
        myClassifier.data_load(npydir) # load training data
        myClassifier.clf_train() # train classifier
        myClassifier.clf_save(coarseclfname, clfdir) # save classifier
        myClassifier.clf_load(coarseclfname, clfdir)
        myClassifier.clf_test() # test classifier


if __name__ == '__main__':
        main()
