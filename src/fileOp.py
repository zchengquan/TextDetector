#!/usr/bin/env python
#title           :fileOp.py
#description     :file loading and saving functions.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from fileOp import * 
#notes           :
#python_version  :2.7

# import modules

import os
import time
import numpy
import pickle
from scipy.misc import imread, imsave, imresize

#------------------------loading and saving ---------------------------------#
def codebook_save(arr, filename, dirname):
	'''codebook_save(arr, filename)

	save codebook into file
	arr: numpy array contains codebook
	filename: file name to save '''
	numpy.save(os.path.join(dirname,filename), arr)
	return

def codebook_load(filename):
	''' codebook_load(codebookname)

	codebookname: _string_ contains the filename of codebook
	Return: array contains codebook '''
	codebook = numpy.load(filename)
	if not len(codebook.shape) == 2:
		raise Exception('Unrecognized codebook shape')
	m, n = codebook.shape
	return numpy.float32(codebook.reshape((m, int(n**0.5), int(n**0.5))))

def npy_save(arr, filename, dirname):
	''' _npy_save(arr, filename)

	save numpy array into npy file
	arr: _numpy_array to save
	filename: _string_ contains target filename for npy file	'''
	make_dir(dirname)
	target = os.path.join(dirname, filename)
	numpy.save(target, arr)
	return

def npy_load(filename, dirname):
	''' _npy_load(mode = 'L')

	load feature or label numpy array 
	Return: _numpy_arrays_ '''
	return numpy.load(os.path.join(dirname, filename))

def data_save(P, L, filename, dirname):
	''' save feature, label pairs into data directory '''
	npy_save(P, 'p_'+filename, dirname)
	npy_save(L, 'l_'+filename, dirname)
	return 

def data_load(dirname):
	''' load feature, label pairs from data direcotry

	dirname: string, data directory name
	Return: tuple of numpy arrays, feature, label pairs. '''
	npylist = os.listdir(dirname)
	estilen = len(npylist)
	L = []
	P = []
	epoch = 0
	for k, featname in enumerate(npylist):
		if k * 10 / estilen > epoch:
			epoch = k*10/estilen
			print 'loading', epoch*10, '%'
		if not featname.startswith('p'):
			continue
		labename = 'l'+featname[1:]
		feat = numpy.load(os.path.join(dirname, featname))
		labe = numpy.load(os.path.join(dirname, labename))
                try:
                        if L == []:
                                P = feat 
                                L = labe
                        else:
                                P = numpy.concatenate((P, feat))
                                L = numpy.concatenate((L, labe))
                except:
                        print featname,
                        print ' numpy array shape does not match ',
                        print feat.shape
	return P, L 

def txt_save(string, filename, dirname):
	'''txt_save(string, dirname, filename)

	save text files
	string: text string to save
	filename: text filename
	dirname: directory name contains file'''
	make_dir(dirname)
	with open(os.path.join(dirname, filename), 'w') as outputfile:
		outputfile.write(string)

def txt_load(filename, dirname):
	'''txt_save(string, dirname, filename)

        load txt files
	filename: text filename
	dirname: directory name contains file'''
	with open(os.path.join(dirname, filename), 'r') as inputfile:
                txtlines = inputfile.read()
                return txtlines

def pickle_save(data, filename, dirname):
        '''pickel_save(data, filename, dirname)

        save data into pickel files
        data: data to be saved
        filename: pickle file name
        dirname: direcotry name '''
        make_dir(dirname)
        if os.path.isfile(os.path.join(dirname, filename)):
                return
        pickle.dump(data, open(os.path.join(dirname, filename), 'wb'))
        return

def pickle_load(filename, dirname):
        import gzip
        ''' pickle_load(filename, dirname)

        load pickle files
        filename: pickle file name
        dirname: direcotry namei
        Return data '''
        try:
                data = pickle.load(open(os.path.join(dirname, filename)))
        except:
                data = pickle.load(gzip.open(os.path.join(dirname, filename)))
        return data

        
def make_dir(dirname):
	''' _make_dir(dirname)
		
	make direcotry if does not exist yet
	dirname: string, check the dir name in string, if not exist, create it
	Return: None '''
	if not os.path.exists(dirname):
		os.makedirs(dirname)

