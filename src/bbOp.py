#!/usr/bin/env python
#title           :bbOp.py
#description     :bounding box information transformation.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from bbOp import * 
#notes           :
#python_version  :2.7

import numpy

#def bbstruct2coor(bbstruct, mode = 'draw'):
#	''' correct bounding box format of ground truth files,
#		 for drawing '''
#	Coor = []
#	if mode == 'draw': # this format is for drawing bouding boxes
#		for bb in bbstruct:
#			coor = bb
#			c = list(coor)
#			c[2] = coor[0] + coor[2]
#			c[3] = coor[1] + coor[3]
#			c = numpy.asarray(coor)
#			Coor.append(c)
#	elif mode == 'label': # this format is for extracting labels of patches 
#		for bb in bbstruct:
#			coor = bb
#			coor = numpy.asarray(coor)
#			Coor.append(coor)
#	return Coor 
#
#def labelStr2coor(label_string):
#	''' correct bounding box format for prediction files,
#		 for drawing '''
#	Coor = []
#	for bb in label_string.split('\r\n'):	
#		if not bb.strip(): # in case that label string contains nothing
#			continue
#		coor = map(int, bb.split(','))
#		c = list(coor)
#		Coor.append(c)
#	return Coor 
	
def coor_resize(coor, r):
	''' coor_resize(self, coor, r)

	resize bounding box information for different scales of image
	coor: list of numpy array contains bounding box coordinates
	r: resize ratio
         :
	Return: list of numpy array contains bounding box coordinates after resize ''' 
	Coor = [] 
	for bb in coor:
                bb = numpy.asarray(bb)
		Coor.append(bb*r)
	return Coor

def imageresize(im, r):
	''' coor_resize(self, coor, r)

	resize bounding box information for different scales of image
	coor: list of numpy array contains bounding box coordinates
	r: resize ratio
         :
	Return: list of numpy array contains bounding box coordinates after resize ''' 
	M = [] 
	for m in im:
                m = numpy.asarray(m)
		M.append(m*r)
	return M

def bb2string(bb):
        '''bb2sting(bb)

        convert bounding box list into strings of ICDAR format
        input: 
                bb, list of list, the coordinates of bounding boxes
        Return:
                ss, string, txt file format, csv.
        '''
        ss = [] 
        for b in bb:
                s = map(lambda x: str(x), b)
                s = ','.join(s)
                ss.append(s)
        ss = '\r\n'.join(ss)
        return ss

def bbenlarge(bb, s, m=50, n=50):
        '''bbenlarge(b, s)

        enlarge bounding box size, but not exceed the image size
        b: list of bounding boxes
        s: list of image shape, get using image.shape
        m: enlarge in x axis
        n: enlarge in y axis
        '''
        newb = []
        for b in bb:
                x0, y0, x1, y1 = b 
                x0 = max(x0 - m, 0)
                y0 = max(y0 - n, 0)
                x1 = min(x1 + m, s[0])
                y1 = min(y1 + n, s[1])
                newb.append([x0, y0, x1, y1])
        return newb 


def normPadding(c, (m, n)):
        '''normPadding(c)

        make the image patch square by addding zeros
        c: original coordinates list
        m: limit of x axis
        n: limit of y axis
        Return: square bounding box coordinates and sizes. 
        '''
        x, y, w, h = c
        if h > w:
                padding = int(h - w)/2
                x1 = max(0, x-padding)
                x2 = min(m, x+w+padding)
                y1 = y
                y2 = y+h
        else:
                padding = int(w - h)/2
                y1 = max(0, y-padding)
                y2 = min(n, y+h+padding)
                x1 = x
                x2 = x+w
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h 

def xywh2xy(c):
        '''xywh2xy(c)

        convert x, y, w, h list into x1, y1, x2, y2 list
        c: original list      
        Return: x1, y1, x2, y2 list
        '''
        x, y, w, h = c
        x2 = x+w        
        y2 = y+h
        return x, y, x2, y2

def xy2xywh(c):
        '''xy2xywh(c)

        convert x1,y1,x2,y2 list into x, y, w, h list
        c: original list      
        Return: x, y, w, h list
        '''
        x, y, x2, y2 = c
        w = x2 - x        
        h = y2 - y
        return x, y, w, h 

def bb2feature(train_feature):
        '''bb2feature(bbfeature)

        convert boudning box information into bounding box features
        Input:
                train_feature, array-like, m by 6 feature vectors, contains x0, y0, width, height,
                image width, image height.
        Output:
                bbfeature, array-like, 19 contains bounding box features computed from the 6,
        '''

        train_feature_x = train_feature[:, 0]
        train_feature_y = train_feature[:, 1]
        train_feature_width = train_feature[:, 2]
        train_feature_height = train_feature[:, 3]
        train_feature_imgwidth = train_feature[:, 4]
        train_feature_imgheight = train_feature[:, 5]

        train_feature_area = train_feature_width * train_feature_height
        train_feature_aspectRatio = numpy.float32(train_feature_width) / numpy.float32(train_feature_height + 1)
        train_feature_imgarea = train_feature_imgwidth * train_feature_imgheight

        train_feature_cx = train_feature_x+train_feature_width/2
        train_feature_cy = train_feature_y+train_feature_height/2
        train_feature_x0 = train_feature_x
        train_feature_y0 = train_feature_y
        train_feature_x1 = train_feature_x + train_feature_width
        train_feature_y1 = train_feature_x + train_feature_height
        # distance to the boundary of the image from x1
        train_feature_x2 = train_feature_imgwidth - train_feature_x1
        # distance to the boundary of the image from y1
        train_feature_y2 = train_feature_imgheight - train_feature_y1 
        train_feature_widthRatio = numpy.float32(train_feature_width) / numpy.float32(train_feature_imgwidth)
        train_feature_heightRatio = numpy.float32(train_feature_height) / numpy.float32(train_feature_imgheight)
        train_feature_areaRatio = numpy.float32(train_feature_area) / numpy.float32(train_feature_imgarea)

        train_feature_x0Ratio = numpy.float32(train_feature_x0) / numpy.float32(train_feature_imgwidth)
        train_feature_y0Ratio = numpy.float32(train_feature_y0) / numpy.float32(train_feature_imgheight)

        train_feature_x1Ratio = numpy.float32(train_feature_x1) / numpy.float32(train_feature_imgwidth)
        train_feature_y1Ratio = numpy.float32(train_feature_y1) / numpy.float32(train_feature_imgheight)

        bbfeature = numpy.asarray([train_feature_width, train_feature_height, 
                train_feature_area, train_feature_aspectRatio, 
                train_feature_cx, train_feature_cy, 
                train_feature_x0, train_feature_y0, 
                train_feature_x1, train_feature_y1, 
                train_feature_x2, train_feature_y2, 
                train_feature_widthRatio, train_feature_heightRatio, train_feature_areaRatio,
                train_feature_x0Ratio, train_feature_y0Ratio,
                train_feature_x1Ratio, train_feature_y1Ratio]).T
        return bbfeature
