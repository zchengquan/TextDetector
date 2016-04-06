#!/usr/bin/env python
#title           :use_ckmean.py
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
import ckmean
import os
#==============================================================================
def changeShape(ft):
        '''changeShape(ft)

	find 8 by 8 patches from 32 by 32 image patches
        Input:
                ft: array like, 32 by 32 by number of samples
        Return:
                array like, number of samples by number of features (64)'''
	m = ft.shape[2]
	data = numpy.zeros((8, 8, 4, m))
	for i in range(m):
		data[:, :, 0, i] = ft[0:8, 0:8, i]
		data[:, :, 1, i] = ft[8:16, 8:16, i]
		data[:, :, 2, i] = ft[16:24, 16:24, i]
		data[:, :, 3, i] = ft[24:32, 24:32, i]
	a1, a2, a3, a4 = data.shape
	data = numpy.reshape(data, (a1*a2, a3*a4))
	data = data.transpose()
	return data

def PL_load(dirname):
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
        if featname.startswith('P'):
            labename = 'L'+featname[1:]
            feat = numpy.load(os.path.join(dirname, featname))
            labe = numpy.load(os.path.join(dirname, labename))
            if L == []:
                P = feat
                L = labe
        else:
                P = numpy.concatenate((P, feat))
                L = numpy.concatenate((L, labe))
    return P, L

def use_ckmean(p, kk = 1000, ii = 100):
	# load image patches
	p = numpy.reshape(p, (p.shape[0], p.shape[1]**0.5, p.shape[1]**0.5))
	data = numpy.transpose(p, (1, 2, 0))
	data = changeShape(data)	
	w = numpy.ones(data.shape[0])
	D = ckmean.init(data, kk)
	D, idx, dm  = ckmean.update(data, D, ii, w)
	numpy.save('codeBook', D)
	numpy.save('codeBookIdx', idx)
	numpy.save('codeBookErr', dm)
	return D, idx, dm

def main():
	
	# load image patches
	kk = 1000
	ii = 100
	data = numpy.load('../../shared_data/syntheticData_img.npy')		
	#data = numpy.load('../../shared_data/SiyuZhuTest.npy')

	data = changeShape(data)	
	w = numpy.ones(data.shape[0])
	D = ckmean.init(data, kk)
	D, idx, dm  = ckmean.update(data, D, ii, w)
	numpy.save('codeBook', D)
	numpy.save('codeBookIdx', idx)
	numpy.save('codeBookErr', dm)

if __name__ == '__main__':
	main()
