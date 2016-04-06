#!/usr/bin/env python
#title           :parseData.py
#description     :parse ground truth data and save into uniform format.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from parseData import *
#notes           :
#python_version  :2.7

# import modules

import os
import numpy
from mail import mail
from imgOp import *
from fileOp import *
from bbOp import *
from timeLog import timeLog

MAIL_REC = 'junesiyu@gmail.com'

#------------------------------Class textDetect2003-------------------------------------#
class parseWord2003:

        def _struct2coor(self, bbstruct):
            COOR = []
            for bb in bbstruct:
                coor = self._get_boundingbox(bb)
                if coor:
                        COOR.append(coor)
            return COOR

        def _struct2label(self, bbstruct, mode = 'train'):
            COOR = []
            for bb in bbstruct:
                if mode == 'train':
                        coor = self._get_charlabel(bb)
                else:
                        coor = self._get_wordlabel(bb)
                if coor:
                        COOR.append(coor)
            return COOR

        def _get_wordlabel(self, bb):
                attrib = bb.strip().split(',')
                if len(attrib) < 4:
                        return
                charlabel = attrib[-1].strip().strip('"')
                return charlabel

        def _get_charlabel(self, bb):
                attrib = bb.strip().split()
                if len(attrib) < 8:
                        return
		row_flag = attrib[0]
		if row_flag[0] == '#':
			return
                charlabel = attrib[-1].strip('"')
                return charlabel

	def _get_boundingbox(self, bb):
		''' _get_boundingbox(self, bb)
		
		read bounding box information from file
		bb: _tuple_ nods in xml file
		Return _tuple_ contains bounding box (x, y, w, h)'''
		attrib = bb.attrib
		x = int(float(attrib['x']))
		y = int(float(attrib['y']))
		w = int(float(attrib['width']))
		h = int(float(attrib['height']))
		return (x, y, w, h)

	def parseData(self, imdir, xmlfilename):
		''' parse_xml(self, imdir, xmlfilename)
		
		parse xml file 
		imdir:_string_ directory name contains images
		xmlfilename:_string_ .xml filename of ground truth
		Return _tuple_ 'root node of xml file '''

                import xml.etree.ElementTree as et
		tree = et.parse(xmlfilename)
		root = tree.getroot()
		dataStack = [] 
		for xml_struct in root:
			data = []
			data.append(xml_struct[0].text)
			data.append(self._struct2coor(xml_struct[2]))
			dataStack.append(data)
		return dataStack

        def prepareImg(self, dataStack, indir, outdir):
                ''' image_pretest(self)

                test and show the ground truth label images from the training and testing data
                before we start the training process.	
                '''
                mytime = timeLog('../timelogs/prepare_img')
                make_dir(outdir)
                # loop for every images in directory
                for k, data in enumerate(dataStack):
                        imgname, coor = data
                        print imgname, str(k), '/', len(dataStack)
                        mytime.start(imgname)
                        img = image_load(imgname, indir)
                        img, coor = imsize_reduce((1400, 1400), img, coor)
                        image_save(img, imgname, outdir)
                        dataStack[k] = (imgname, coor)
                        mytime.end()
                mail(MAIL_REC, '<Coding Message!!!>, image_pretest_finished')
                mytime.final('prepare images finished.')
                return dataStack

#--------------------------------Class parseWord2013-------------------------------#
class parseWord2013(parseWord2003):

	def _get_bbstruct(self, filename, bbdir):
		''' _get_bbstruct(self, filename)

		get corresponding bb file for image
		filename: _string_ 
		Return _list_ contains file content '''
		bbname = os.path.join(bbdir,
			 ('gt_'+filename.split('.')[0] + '.txt'))
		return open(bbname, 'r').readlines()

	def _get_boundingbox(self, bb):
		'''_get_boundingbox(self, bb)

		read bounding box information from file
		bb: nods in xml file
		Return: _tuple_ contains bounding box (x, y, w, h)'''
		attrib = bb.strip().split(',')
		x = int(float(attrib[0].strip()))
		y = int(float(attrib[1].strip()))
		w = int(float(attrib[2].strip()))
		h = int(float(attrib[3].strip()))
		w = w - x
		h = h - y
		return (x, y, w, h)

	def parseData(self, imdir, bbdir):
		''' parse_xml(self, imdir, bbdir)

		load image and boundingbox files 
		imdir: _string_ contains directory for images
		bbdir: _string_ contains directory for boundingbox files '''

		dirlist = os.listdir(imdir)
		dataStack= [] 
		for filename in dirlist:
			data = []
			data.append(filename)
			data.append(self._struct2coor(self._get_bbstruct(filename, bbdir)))
			dataStack.append(data)
                return dataStack

        def parseLabel(self, imdir, bbdir, mode = 'train'):
		''' parseLabel(self, imdir, bbdir)

		load image and boundingbox files 
		imdir: _string_ contains directory for images
		bbdir: _string_ contains directory for boundingbox files '''
		dirlist = os.listdir(imdir)
		dataStack= [] 
		for filename in dirlist:
			data = []
			data.append(filename)
			data.append(self._struct2label(self._get_bbstruct(filename, bbdir), mode = mode))
			dataStack.append(data)
                return dataStack
#---------------------------------------END---------------------------------------------#

#-----------------------------Class parseChar2013----------------------------------#
class parseChar2013(parseWord2013):

	def _get_bbstruct(self, filename, bbdir):
		''' _get_bbstruct(self, filename)

		get corresponding bb file for image
		filename: _string_ 
		return: _list_ contains file content '''
		# from image file name to get corresponding bb file
		bbname = os.path.join(bbdir, (filename.split('.')[0] + '_GT.txt'))
		return open(bbname, 'r').readlines()

	def _get_boundingbox(self, bb):
		'''_get_boundingbox(self, bb)

		read bounding box information from file
		bb: _tuple_ nods in xml file
		return: _tuple_ contains bounding box (x, y, w, h)'''
                attrib = bb.strip().split()
		if len(attrib) < 8:
			return 
		row_flag = attrib[0]
		if row_flag[0] == '#':
			return
		x = int(float(attrib[5]))
		y = int(float(attrib[6]))
		w = int(float(attrib[7]))
		h = int(float(attrib[8]))
		w = w - x 
		h = h - y
		return (x, y, w, h)

