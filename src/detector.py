#!/usr/bin/env python
#title           :detector.py
#description     :Class for text detector.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from detector import detector
#notes           :
#python_version  :2.7

# import modules
import pdb
import os
import time
import numpy
import sys
sys.path.insert(0, '/usr/lib/python2.7/dist-packages')
import cv2

from sys import argv
from joblib import Parallel, delayed

from featExt import featExt, get_patch
from para_classifier import para_adaboost
from cluster.use_ckmean import use_ckmean
from mail import mail
import plot
from arrayOp import *
from bbOp import *
from fileOp import *
from imgOp import *
from timeLog import timeLog
from skimage.transform import rotate


# No. of Jobs for parallel computing

#---------------------------------------END---------------------------------------------#
class detector:

	def __init__(self, codebook, data,
                psize = 32, ssize = 16, nob = 3,
                rpower = 0.9 ** numpy.asarray(range(30)),
                para0 = 0.8, para1 = 0.3,
                labelMode = 'foreground', featMode = 'local', edgeMode = True):

		self._posCount = numpy.zeros(len(rpower)) # positive sample count
                self.codebook = codebook
                self.data = data
                self.psize = psize
                self.ssize = ssize
                self.rpower = rpower
                self.para0 = para0
                self.para1 = para1
                self.nob = nob
		self.labelMode = labelMode
		self.featureMode = featMode
                self.edgeMode = edgeMode
                self.noofjobs = 20
                self.mail_rec = 'junesiyu@gmail.com'
#--------------------------------extracting features------------------------------------#

        def set_featureMode(self, featMode = 'local'):
                self.featureMode = featMode
                return

        def set_codebook(self, codebook):
               self.codebook = codebook
               return 

	def width_height_comp(self, bb, psize, para):
		'''width_height_comp(bb, psize, para)

		compare bounding box width and height with patch size,
		if patch size is widthin (1-para) to (1+para) range of bounding box width or height
		returns true, otherwise, returns false
		bb: list of bounding box, x, y, w, h
		psize: current patch size
		para: threshold of patch size and bounding box size difference
		Return: boolean '''
		_, _, w, h = bb
		if ((psize < w*(para+1) and psize > w*(1-para))
                        or (psize < h*(para+1) and psize > h*(1-para))):
			return True
		return False

	def area_comp(self, bb1, bb2, para):
		'''area_comp(bb, bb2, para)

		compare two bounding box overlapping area,
		if overlapping area is larger than threshold para, returns true,
		otherwise, returns false
		bb: list of bounding box, x, y, w, h
		psize: patch bounding box, i, j, psize, psize
		para: threshold of overlapping area of patch and bounding box
		Return: boolean '''

		x, y, w, h = bb1
		i, j, psize, _ = bb2
                a = (min(x+w, i+psize) - max(x, i))
		b = (min(y+h, j+psize) - max(y, j))
		if (a > 0 and b > 0 and a*b > psize * psize * para):
			return True
                return False

	def image2label(self, img, coor,
                psize, ssize, para0, para1,
                target = 'foreground'):
		''' image2label(img, coor, psize, ssize, para0, para1)

		generate label for corresponding patches
		labels are corresonding to patches get from self._image2feat
		img: 2D_numpy_array
		Return 2D_numpy_array'''
		(ind1, ind2) = map(lambda var:range(0, int(var)-psize, ssize), img.shape)
		label = numpy.zeros((len(ind1), len(ind2)))
		for bb in coor:
			# Comparing patch size:
                        if not self.width_height_comp(bb, psize, para1):
				continue
			# Comparing patch location (overlapping):
			for p, i in enumerate(ind1):
				for q, j in enumerate(ind2):
					if self.area_comp(bb, [i, j, psize, psize], para0):
						label[p, q] = 1
		if target == 'foreground':
			return label
		if target == 'whitespace':
			kernel = numpy.ones((3, 3), dtype = numpy.float32)
			labelb = cv2.dilate(label, kernel, iterations = 1) - label
			return labelb
		if target == 'whitespace_strict':
			kernel = numpy.ones((3, 1), dtype = numpy.float32)
			labelb = cv2.morphologyEx(label, cv2.MORPH_CLOSE, kernel) - label
			return labelb
                else:
                        raise Exception('Target '+target+' is not found!')
                        return

	def label2image(self, img, label, mode = 'block'):
		''' _label2image(img, label, psize)

		convert label into images, label could be ground truth or prediction labels
		img: 2D_numpy_array
		label: 2D_numpy_array
		Return 2D_numpy_array'''
                psize = self.psize
                ssize = self.ssize
		limg = numpy.zeros(img.shape, dtype = numpy.float32) # label image
		(ind1, ind2) = map(lambda var:range(0, int(var)-psize, ssize), img.shape)
                if mode == 'block':
                        for p, i in enumerate(ind1):
                                for q, j in enumerate(ind2):
                                        limg[i:i+psize, j:j+psize] = numpy.maximum(label[p, q],
                                                limg[i:i+psize, j:j+psize])
                elif mode == 'point':
                        for p, i in enumerate(ind1):
                                for q, j in enumerate(ind2):
                                        limg[i+psize/2, j+psize/2] = numpy.maximum(label[p, q],
                                                limg[i+psize/2, j+psize/2])
                limg = limg - numpy.min(limg)
                limg = limg / (numpy.max(limg) + 0.0001)
		return numpy.uint8(limg*255)

	def image2feat(self, img, psize, ssize, nob = 3,
                codebook = None, target = 'local', edgeMode = True):
		'''_image2feat(self, img, r, mode = 'test')

		convert image into convolutional features
		img: 2D_numpy_array with size (w, h)
		r: _float_ resize ratio of the patch comparing to original psize0
		Return: patch _numpy_array_ with shape (m*n, psize0*psize0)
			label array with shape (m*n)'''
                if edgeMode:
			img = numpy.float32(img) # data type!!
			img = image2edge(img)# compute edge map
                '''
                if whitenMode:
                        img =
                '''
		if target == 'local' or target == 'context': # use parallel processing
			convImg = Parallel(n_jobs = self.noofjobs)(
                                delayed(featExt)(img, codeim, (psize, ssize), nob, mode = target)
                                for codeim in codebook )
			patch_conv = numpy.empty((convImg[0].shape[0],
				convImg[0].shape[1],
				convImg[0].shape[2],
				convImg[0].shape[3],
				len(convImg)), dtype = numpy.uint8)
			for k, patch in enumerate(convImg):
				patch_conv[:, :, :, :, k] = patch

		elif target == 'original': # no convolution
			patch = get_patch(img, psize, ssize)
			patch_conv = numpy.empty((patch.shape[0],
                                patch.shape[1],
                                patch.shape[2],
                                patch.shape[3], 1))
			patch_conv[:, :, :, :, 0] = patch

		return patch_conv

        def feat2pred(self, img, patch, clf):
                ''' feat2pred(img, patch, clf)

                convert feature patches into predictions
                patch: feature array
                clf: classifier
                Return: prediction image '''
                m, n, _, _, _ = patch.shape
                patch = flatten_patch(patch)
                pred = clf.decision_function(patch)
                pred = pred_norm(pred)
                pred = pred.reshape((m, n))
                return pred

        def multiscale_label_ext(self, img0, coor0):
		'''  _multiscale_label_ext(self,img0, coor0)

		convert image to label image to pre-qualify the ground truth information
		img: image numpy array with shape (m, n)
		Return: Label image LABEL '''

		# process in different scales
		label_stack = []
		for l, r in enumerate(self.rpower):
			try:
				img = img_resize(img0, r)
				coor = coor_resize(coor0, r)
				label = self.image2label(img, coor,
                                        self.psize, self.ssize,
                                        self.para0, self.para1,
                                        target = self.labelMode)
				label_stack.append(label)
				self._posCount[l] += int(numpy.sum(label))
			except:
				continue
		# return label image and bounding box strings
		print 'Positive samples Count', numpy.sum(self._posCount)
		return label_stack

	def multiscale_patch_ext(self, img0, coor0, mode = 'train'):
		'''  _multiscale_train(self,img)

		convert image to feature patches in multiscales
		img: image numpy array with shape (m, n)
		Return P: 2D_numpy_array with shape m*n, by No.-of-features
                L: 1D_numpy_array corresponding label with shape m*n by 1'''

		patch_stack = []
		label_stack = []
		for l, r in enumerate(self.rpower):
                        img = img_resize(img0, r)
                        coor = coor_resize(coor0, r)
                        label = self.image2label(img, coor,
                                self.psize, self.ssize,
                                self.para0, self.para1,
                                target = self.labelMode)
                        # if no foregournd labels are found in the corrent scale
                        # then don't extract patch
                        if numpy.sum(label) == 0 and mode == 'train':
                            continue
                        patch = self.image2feat(img,
                                self.psize, self.ssize,
                                self.nob, self.codebook,
                                target = self.featureMode,
                                edgeMode = self.edgeMode)
                        label_stack.append(label)
                        patch_stack.append(patch)
                        self._posCount[l] += int(numpy.sum(label))
		print 'Positive samples Count', numpy.sum(self._posCount)
		return patch_stack, label_stack


	def multiscale_test(self, img0, clf):
		predpnt_stack = []
                predblk_stack = []
		for k, r in enumerate(self.rpower):
                        img = img_resize(img0, r)
                        patch = self.image2feat(img,
                                self.psize, self.ssize,
                                self.nob, self.codebook,
                                target = self.featureMode,
                                edgeMode = self.edgeMode)
                        pred = self.feat2pred(img, patch, clf)
                        predpnt = self.label2image(img, pred, mode = 'point')
                        predblk = self.label2image(img, pred, mode = 'block')
                        predpnt_stack.append(predpnt)
                        predblk_stack.append(predblk)
		return predpnt_stack, predblk_stack


        def multiscale_test2(self, img0, image0):
		predpnt_stack = []
                predblk_stack = []
		for k, r in enumerate(self.rpower):
                        try:
                                img = img_resize(img0, r)
                                edge_img = image2edge(img)
                                image = imageresize(image0, r)
                                label = self.image2label(img, image,
                                            self.psize, self.ssize,
                                            self.para0, self.para1,
                                            target = self.labelMode)
                                predpnt = self.label2image(img, label, mode = 'point')
                                predblk = self.label2image(img, label, mode = 'block')
                                predpnt_stack.append(predpnt)
                                predblk_stack.append(predblk)
                        except:
                                continue
		return predpnt_stack, predblk_stack

	def image_train(self, indir, outdir):
		''' image_train(self)

		convert image to patch for all images in imdir
		'''
		# loop for every images in directory
                mytime = timeLog('../timelogs/image_train')
                make_dir(outdir)
		for k, data in enumerate(self.data):
                        try:
                                print 'Image-Train:', data[0], k,'/',len(self.data)
                                imgname = data[0]
                                coor = data[1]
                                img = image_load(imgname, indir)
                                mytime.start(imgname)
                                patch_stack, label_stack = self.multiscale_patch_ext(img, coor)
                                patch_stack, label_stack = stack2array(patch_stack, label_stack)
                                data_save(patch_stack, label_stack,
                                        os.path.splitext(imgname)[0], outdir)
                        except:
                                continue
                        finally:
                                mytime.end()
		mail(self.mail_rec, '<Coding Message!!!>, image_train_finished')
                mytime.final('image train finished')
		return


	def image_test(self, indir, outdir, clf):
		''' image_train(self)

		convert image to patch for all images in imdir
		'''
                self.imageprocess(indir, outdir, clf)
                mytime = timeLog('../timelogs/image_test')
                make_dir(outdir)
                # loop for every images in directory
                if not 'decision_function' in dir(clf):
                        clf.decision_function = clf.predict
                        print 'decision function created!'
                for k, data in enumerate(self.data):
                        print 'Image-Test:', data[0], k,'/',len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)
                        img = image_load(imgname, indir)
                        res_stack = self.multiscale_test(img, clf)
                        pickle_save(res_stack, os.path.splitext(imgname)[0], outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, image_test_finished!')
                mytime.final('image test finished')
		return

	def imageprocess(self, indir, outdir, clf):
                mytime = timeLog('../timelogs/image_test')
                make_dir(outdir)
                # loop for every images in directory
                if not 'decision_function' in dir(clf):
                        clf.decision_function = clf.predict
                        print 'decision function created!'
                for k, data in enumerate(self.data):
                        print 'Image-Process:', data[0], k,'/',len(self.data)
                        imgname = data[0]
                        image = data[1]
                        mytime.start(imgname)
                        img = image_load(imgname, indir)
                        res_stack = self.multiscale_test2(img, image)
                        pickle_save(res_stack, os.path.splitext(imgname)[0], outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, image_test_finished!')
                mytime.final('image test finished')
		return

        def geneHotmap(self, indir, outdir):
                '''geneHotmap(self, indir, outdir)

                convert stacks from image_test to hotmaps
                '''
                mytime = timeLog('../timelogs/testHotmap')
                make_dir(outdir)
                for k, data in enumerate(self.data):
                        try:
                                print 'Generate hotmaps:', data[0], k, '/', len(self.data)
                                imgname = data[0]
                                mytime.start(imgname)
                                imgname = os.path.splitext(imgname)[0]
                                s = pickle_load(imgname, indir)
                                pnt = stack2image(s[0])
                                blk = stack2image(s[1])
                                image_save(pnt, 'point'+imgname+'.png', outdir)
                                image_save(blk, 'block'+imgname+'.png', outdir)
                        except:
                                continue
                        finally:
                                mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, hotmap generation finished')
                mytime.final('hotmap generation finished')
                return

        def geneTxt(self, indir, outdir, thre = 128):
                '''geneTxt(self, indir, outdir)

                generate csv file from the hotmaps to be in ICDAR standard format
                '''
                mytime = timeLog('../timelogs/testTxtFile')
                make_dir(outdir)
                for k, data in enumerate(self.data):
                        print 'Generate txt files:', data[0], k, '/', len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)

                        basename = os.path.splitext(imgname)[0]
                        blkname = 'block'+basename+'.png' # use block plot as hotmap
                        txtname = 'res_img_'+basename+'.txt'

                        img = image_load(blkname, indir)
                        bb = image2bb(img, thre)
                        s = bb2string(bb)
                        txt_save(s, txtname, outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, txt generation finished')
                mytime.final('txt file generation finished')
                return

        def geneROI(self, imdir, indir, outdir, thre = 128):
                ''' geneROI(self, indir, outdir)

                crop the region of interest from the hotmap
                '''
                mytime = timeLog('../timelogs/regionOfInterest')
                make_dir(outdir)
                for k, data in enumerate(self.data):
                        print 'Generate region of interests:', data[0], k, '/', len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)

                        basename = os.path.splitext(imgname)[0]
                        blkname = 'block' + basename + '.png' 
                        raw = image_load(imgname, imdir)
                        img = image_load(blkname, indir)
                        bb = image2bb(img, thre)
                        bb = bbenlarge(bb, img.shape)
                        roi_stack = image2roi(raw, bb)
                        pickle_save(roi_stack, basename, outdir)

                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, roi generation finished')
                mytime.final('roi generation finished')
                return

        
        def geneROI2(self, imdir, indir, outdir, thre = 128):
                ''' geneROI2(self, indir, outdir)

                crop the region of interest from the stack
                '''
                
                mytime = timeLog('../timelogs/testHotmap')
                make_dir(outdir)
                for k, data in enumerate(self.data):
                        print 'Generate roi:', data[0], k, '/', len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)
                        imgname = os.path.splitext(imgname)[0]
                        s = pickle_load(imgname, indir)[1]
                        roi_stack = []
                        for layer in s:
                                bb = image2bb(img, thre)
                                bb = bbenlarge(bb, img.shape)
                                roi_stack.append(image2roi(raw, bb))
                        pickle_save(roi_stack, basename, outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, hotmap generation finished')
                mytime.final('hotmap generation finished')
                return

        def roi_test2(self, indir, outdir, clf):
		''' roi_test2(self, indir, outdir, clf)

                apply detection on region of interest
		'''
                mytime = timeLog('../timelogs/image_test')
                make_dir(outdir)
                # loop for every images in directory
                if not 'decision_function' in dir(clf):
                        clf.decision_function = clf.predict
                        print 'decision function created!'

                for k, data in enumerate(self.data):
                        print 'Image-Test:', data[0], k,'/',len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)
                        roiname = os.path.splitext(imgname)[0]
                        roistack = pickle_load(roiname, indir)

                        predpnt_stack = []
                        predblk_stack = []
                        for img in roistack:
                                patch = self.image2feat(img,
                                        self.psize, self.ssize,
                                        self.nob, self.codebook,
                                        target = self.featureMode,
                                        edgeMode = self.edgeMode)
                                pred = self.feat2pred(img, patch, clf)
                                predpnt = self.label2image(img, pred, mode = 'point')
                                predblk = self.label2image(img, pred, mode = 'block')
                                predpnt_stack.append(predpnt)
                                predblk_stack.append(predblk)

                        pickle_save((predpnt_stack, predblk_stack), os.path.splitext(imgname)[0], outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, image_test_finished!')
                mytime.final('image test finished')
		return

        def roi_test(self, indir, rawdir, outdir, clf):
		''' roi_test(self, indir, outdir, clf)

                apply detection on region of interest
		'''
                mytime = timeLog('../timelogs/image_test')
                make_dir(outdir)
                # loop for every images in directory
                if not 'decision_function' in dir(clf):
                        clf.decision_function = clf.predict
                        print 'decision function created!'

                for k, data in enumerate(self.data):
                        try:
                                print 'ROI test:', data[0], k, '/', len(self.data)
                                imgname = data[0]
                                img = data[2] 
                                mytime.start(imgname)
                                raw = image_load(imgname, rawdir)
                                stackname = os.path.splitext(imgname)[0]
                                s = pickle_load(stackname, indir)[1]
                                layer_stack = []
                                for layer in s:
                                        bb = image2bb(layer)
                                        #bb = bbenlarge(bb, layer.shape)
                                        raw = img_resize(raw, layer.shape)
                                        rois = image2roi(raw, bb)
                                        predpnt_stack = []
                                        #predblk_stack = []
                                        for rimg00 in rois:
                                                rimg0 = imreduce(64, rimg00)
                                                predpnt = numpy.zeros_like(rimg0)
                                                for ang in [-6, -4, -2, 0, 2, 4, 6]:
                                                        rimg = rotate(rimg0, ang)
                                                        patch = self.image2feat(rimg,
                                                                self.psize, self.ssize,
                                                                self.nob, self.codebook,
                                                                target = self.featureMode,
                                                                edgeMode = self.edgeMode)
                                                        pred = self.feat2pred(rimg, patch, clf)
                                                        predpnt0 = self.label2image(rimg, pred, mode = 'point')
                                                        predpnt = predpnt + rotate(predpnt0, -ang)
                                                for asp in [0.6, 0.8, 1, 1.2, 1.4]:
                                                        rimg = img_resize(rimg0, [int(rimg0.shape[0]*asp), rimg0.shape[1]])
                                                        patch = self.image2feat(rimg,
                                                                self.psize, self.ssize,
                                                                self.nob, self.codebook,
                                                                target = self.featureMode,
                                                                edgeMode = self.edgeMode)
                                                        pred = self.feat2pred(rimg, patch, clf)
                                                        predpnt0 = self.label2image(rimg, pred, mode = 'point')
                                                        predpnt = predpnt+img_resize(predpnt0, rimg0.shape)
                                                predcc = pnt2cc(rimg0, predpnt)
                                                predcc = img_resize(predcc, rimg00.shape)
                                                #predblk = self.label2image(img, pred, mode = 'block')
                                                predpnt_stack.append(predcc)
                                                #predblk_stack.append(predblk)
                                        layer_stack.append(roi2image(predpnt_stack, bb, layer.shape))
                                img = stack2image(layer_stack, img)
                                pickle_save(layer_stack, stackname, outdir)
                                image_save(img, imgname.split('.')[0]+'.png', outdir)
                                mytime.end()
                        except:
                                mytime.end()
                                continue
                mail(self.mail_rec, '<Coding Message!!!>, hotmap generation finished')
                mytime.final('hotmap generation finished')
                return

        def seed2image(self, roidir, imdir, rawdir, outdir):
                '''seed2image(self, indir, outdir)

                convert roi detection to image 
                '''

                mytime = timeLog('../timelogs/regionOfInterest')
                make_dir(outdir)
                for k, data in enumerate(self.data):
                        print 'Generate region of interests:', data[0], k, '/', len(self.data)
                        imgname = data[0]
                        mytime.start(imgname)
                        basename = os.path.splitext(imgname)[0]
                        blkname = 'block' + basename + '.png' # use block plot as hotmap
                        roi = pickle_load(os.path.splitext(imgname)[0], roidir)
                        raw = image_load(imgname, rawdir)
                        img = image_load(blkname, imdir)
                        bb = image2bb(img)
                        bb = bbenlarge(bb, img.shape)
                        img = roi2image(roi, bb, img.shape)
                        image_save(img, imgname, outdir)
                        mytime.end()
                mail(self.mail_rec, '<Coding Message!!!>, roi generation finished')
                mytime.final('roi generation finished')
                return


