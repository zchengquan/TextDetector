#!/usr/bin/env python
#title           :wordGraph.py
#description     :functions for image operations
#author          :siyu zhu
#date            :Feb, 2015
#version         :0.1
#usage		 :from wordGraph import * 
#notes           :
#python_version  :2.7

# import modules

import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2

import numpy
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from parseData import parseWord2003, parseWord2013, parseChar2013
from imgOp import image_load, imShow
from skimage.measure import label
from sklearn.preprocessing import Imputer
from scipy.sparse.csgraph import connected_components
import pickle
from fileOp import make_dir, txt_save
from zipFile import zipFile

def SMOTE(T, N, k):
        """
        SMOTE(T, N, k)

        Populate samples with SMOTE algorithm
        Input:
        T: array-like, shape = [n_minority_samples, n_features]
           Holds the minority samples
        N: percetange of new synthetic samples: 
           n_synthetic_samples = N/100 * n_minority_samples.
        k: int. Number of nearest neighbours. 

        Returns
        array, shape = [(N/100) * n_minority_samples, n_features]"""
    
        import numpy as np
        from random import randrange, choice
        from sklearn.neighbors import NearestNeighbors

        n_minority_samples, n_features = T.shape
    
        if N < 100:
                N = 100
                pass

        if (N % 100) != 0:
                raise ValueError("N must be < 100 or multiple of 100")
    
        N = N/100
        n_synthetic_samples = N * n_minority_samples
        S = np.zeros(shape=(n_synthetic_samples, n_features))

        #Learn nearest neighbours
        neigh = NearestNeighbors(n_neighbors = k)
        neigh.fit(T)

        #Calculate synthetic samples
        for i in xrange(n_minority_samples):
                nn = neigh.kneighbors(T[i], return_distance=False)
                for n in xrange(N):
                        nn_index = choice(nn[0])
                        #NOTE: nn includes T[i], we don't want to select it 
                        while nn_index == i:
                                nn_index = choice(nn[0])
                
                dif = T[nn_index] - T[i]
                gap = np.random.random()
                S[n + i * N, :] = T[i,:] + gap * dif[:]
        return S


def feature_extraction(train_feature):
        ''' feature_extraction(train_feature)

        convert bounding box information into pair-wise ralationship features, 
        include distance of centers, differnces in width, heights etc. 
        Input:
        train_feature: array-like, n by 2 by 4, n is the number of samples, 
        Return:
        array-like, n by 13 '''
        train_feature = numpy.asarray(train_feature)
        train_feature_width = train_feature[:, :, 2]
        train_feature_height = train_feature[:, :, 3]
        train_feature_area = train_feature[:, :, 2] * train_feature[:, :, 3]
        train_feature_aspectRatio = numpy.float32(train_feature_width) / numpy.maximum(numpy.ones_like(numpy.float32(train_feature_height)), numpy.float32(train_feature_height))
        train_feature_x = train_feature[:, :, 0]
        train_feature_y = train_feature[:, :, 1]
        train_feature_cx = train_feature_x + train_feature_width/2
        train_feature_cy = train_feature_y + train_feature_height/2

        edge_width_dif = abs(train_feature_width[:, 0] - train_feature_width[:, 1])
        edge_height_dif = abs(train_feature_height[:, 0] - train_feature_height[:, 1])
        edge_area_dif = abs(train_feature_area[:, 0] - train_feature_area[:, 1])
        edge_aspectRatio_dif = abs(train_feature_aspectRatio[:, 0] - train_feature_aspectRatio[:, 1])
        edge_cx_dis = abs(train_feature_cx[:, 0] - train_feature_cx[:, 1])
        edge_cy_dis = abs(train_feature_cy[:, 0] - train_feature_cy[:, 1])

        edge_width_mean = abs(train_feature_width[:, 0] + train_feature_width[:, 1]) / 2
        edge_height_mean = abs(train_feature_height[:, 0] + train_feature_height[:, 1]) / 2
        edge_area_mean = abs(train_feature_area[:, 0] + train_feature_area[:, 1]) / 2
        edge_aspectRatio_mean = abs(train_feature_aspectRatio[:, 0] + train_feature_aspectRatio[:, 1]) / 2

        edge_cu_dis = numpy.power(numpy.power(edge_cx_dis,2)+ numpy.power(edge_cy_dis, 2),0.5) # center Euclidean distance
        edge_bd_dis = numpy.maximum(edge_cx_dis - edge_width_mean, edge_cy_dis - edge_height_mean)# closest boundary distance

        edge_cu_dis_norm = numpy.float32(edge_cu_dis) / numpy.float32(edge_area_mean+1)
        edge_bd_dis_norm = numpy.float32(edge_bd_dis) / numpy.float32(edge_area_mean+1)

        edge_width_dif_norm = numpy.float32(edge_width_dif) / numpy.float32(edge_width_mean+1)
        edge_height_dif_norm = numpy.float32(edge_height_dif) / numpy.float32(edge_height_mean+1)

        edge_cx_dis_norm = numpy.float32(edge_cx_dis) / numpy.float32(edge_width_mean+1)
        edge_cy_dis_norm = numpy.float32(edge_cy_dis) / numpy.float32(edge_height_mean+1)

        edge_angle_dis = numpy.arctan(numpy.float32(edge_cx_dis) / numpy.float32(edge_cy_dis+1))

        feature = numpy.asarray([edge_cu_dis, edge_cu_dis_norm, edge_bd_dis, edge_bd_dis_norm, edge_cx_dis, edge_cx_dis_norm,
                        edge_cy_dis, edge_cy_dis_norm, edge_width_dif, edge_width_dif_norm, 
                        edge_height_dif, edge_height_dif_norm, edge_angle_dis]).T
        imputer = Imputer()
        feature = imputer.fit_transform(feature)

        return feature

def minimum_spanning_tree(X, copy_X=True):
        '''minimum_spanning_tree(X, copy_X=True)

        Generate MST for a list of points
        Input:
                X: array like, containing center point coordinates
        Return:
                Array like, index for points that edges between them are in MST. '''
    
        if copy_X:
                X = X.copy()
        if X.shape[0] != X.shape[1]:
                raise ValueError("X needs to be square matrix of edge weights")
        n_vertices = X.shape[0]
        spanning_edges = []
        # initialize with node 0:
        visited_vertices = [0]
        num_visited = 1
        # exclude self connections:
        diag_indices = numpy.arange(n_vertices)
        X[diag_indices, diag_indices] = numpy.inf
        while num_visited != n_vertices:
                new_edge = numpy.argmin(X[visited_vertices], axis=None)
                # 2d encoding of new_edge from flat, get correct indices
                new_edge = divmod(new_edge, n_vertices)
                new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
                # add edge to tree
                spanning_edges.append(new_edge)
                visited_vertices.append(new_edge[1])
                # remove all edges inside current tree
                X[visited_vertices, new_edge[1]] = numpy.inf
                X[new_edge[1], visited_vertices] = numpy.inf
                num_visited += 1
        return numpy.vstack(spanning_edges)

def bimg2bblist(predimg):
        '''bimg2bblist(predimg)

        Convert binary image into character bounding box files
        Input:
                predimg: binary image, indicate foreground or background
        Output:
                bblist1: numpy array, containing char bounding boxes, with format x, y, w, h.'''
        labelimg = label(predimg, background = 0)
        imglabel = numpy.unique(labelimg)
        bblist1 = []
        for i in imglabel:
                if i == -1:
                        continue
                cimg = labelimg==i
                xx, yy = numpy.where(cimg)
                x = min(xx)
                w = max(xx) - x
                y = min(yy)
                h = max(yy) - y

                x1 = numpy.arange(x, x+w)
                y1 = numpy.ones(len(x1)) * y
                x2 = numpy.arange(x, x+w)
                y2 = numpy.ones(len(x2)) * (y+h)

                y3 = numpy.arange(y, y+h)
                x3 = numpy.ones(len(y3)) * x
                y4 = numpy.arange(y, y+h)
                x4 = numpy.ones(len(y4)) * (x+w)

                bblist1.append([x, y, w, h])
        bblist1 = numpy.asarray(bblist1)
        return bblist1

def bblist2relationMat(bblist1, classifier):
        '''bblist2relationMat(bblist1)
    
        Compute relationship matrix (sparse graph) relationMat using character bounding boxes
        Input:
                bblist1: char bounding boxes, with x, y, w, h format
                classifier: segmentation classifier object.
        Output:
                relationMat, number array, each element at position i, j indicate ith and jth char 
                are within the same word (True or False)'''
    
        train_feature = []
        for i in range(len(bblist1)):
                for j in range(i, len(bblist1)):
                        if i != j:
                                train_feature.append([bblist1[i], bblist1[j]])

        feature = feature_extraction(train_feature)    
        pred = classifier.predict(feature)
        relationMat = numpy.zeros((len(bblist1), len(bblist1)))
        k = 0
        for i in range(len(bblist1)):
                for j in range(i, len(bblist1)):
                        if i == j:
                                continue
                        if pred[k]:
                                relationMat[i, j] = 1
                                relationMat[j, i] = 1
                        k += 1
        return relationMat


def relationMat2wordbb(bblist1, relationMat):
        ''' relationMat2wordbb(bblist1, relationMat)
    
        Convert character bounding boxes bblist1 into word bounding boxes wordbb,
        using sparse graph
        Input:
                bblist1, character bounding boxes, each row represents a char, 
                        with x, y, w, h format
                relationMat, sparse graph get from bblist2relationMat,
                        each element indicate True or False
                        that corresponding char belong to the same word
        Output:
                wordbb, word bounding boxes, each row represents a word, with x, y, w, h format.
        '''
        ncc, labelcc = connected_components(relationMat)
        cclabel = numpy.unique(labelcc)
        wordbb = []
        for i in cclabel:
                wlist = bblist1[labelcc == i, ...]
                wlist[:, 2] = wlist[:, 0] + wlist[:, 2]
                wlist[:, 3] = wlist[:, 1] + wlist[:, 3]
                x = min(wlist[:, 0])
                y = min(wlist[:, 1])
                w = max(wlist[:, 2]) - x
                h = max(wlist[:, 3]) - y
                wordbb.append([x, y, w, h])
        return wordbb

def plotbb(predimg, charbb, wordbb):
        fig = plt.figure(frameon = False)
        fig.set_size_inches(10, 10)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(predimg.T)
        plt.axis('off')
    
        for bb in charbb:
                x, y, w, h = bb
                x1 = numpy.arange(x, x+w)
                y1 = numpy.ones(len(x1)) * y
                x2 = numpy.arange(x, x+w)
                y2 = numpy.ones(len(x2)) * (y+h)

                y3 = numpy.arange(y, y+h)
                x3 = numpy.ones(len(y3)) * x
                y4 = numpy.arange(y, y+h)
                x4 = numpy.ones(len(y4)) * (x+w)

                plt.plot(x1, y1, 'g')
                plt.plot(x2, y2, 'g')
                plt.plot(x3, y3, 'g')
                plt.plot(x4, y4, 'g')
    
        for bb in wordbb:
                x, y, w, h = bb
                x1 = numpy.arange(x, x+w)
                y1 = numpy.ones(len(x1)) * y
                x2 = numpy.arange(x, x+w)
                y2 = numpy.ones(len(x2)) * (y+h)

                y3 = numpy.arange(y, y+h)
                x3 = numpy.ones(len(y3)) * x
                y4 = numpy.arange(y, y+h)
                x4 = numpy.ones(len(y4)) * (x+w)

                plt.plot(x1, y1, 'r')
                plt.plot(x2, y2, 'r')
                plt.plot(x3, y3, 'r')
                plt.plot(x4, y4, 'r')
        
        plt.show()
        return

def wordbbMerge(wordbb):
        ratio = 0.5
        newwordbb = []
        for i in range(len(wordbb)):
                x1, y1, w1, h1 = wordbb[i]
                area1 = w1 * h1
                for j in range(len(wordbb)):
                        if i == j:
                                continue
                        x2, y2, w2, h2 = wordbb[j]
                        a = min(x1+w1, x2+w2) - max(x1, x2)
                        b = min(y1+h1, y2+h2) - max(y1, y2)
                        if (a > 0 and b > 0 and a*b > area1 * ratio):
                                break
                else:
                        newwordbb.append(wordbb[i])
        return newwordbb

def wordbb2pred(wordbb, outdir):
        make_dir(outdir)
        for imgname, bb in wordbb:
                print imgname
                txtname = 'res_'+imgname.split('.')[0]+'.txt'
                ss = []
                bb = numpy.asarray(bb)
                bb[:, 2] = bb[:, 0]+bb[:, 2]
                bb[:, 3] = bb[:, 1]+bb[:, 3]
                for b in bb:
                        s = ','.join(map(str, b))
                        ss.append(s)
                ss = '\r\n'.join(ss)
                txt_save(ss, txtname, outdir)
        zipFile('pred.zip', '../data/myexpe/pretxt/')
        return 

def wordGraph_test(pimgdir, classifier):
        '''
        classifier = pickle.load(open('randomForest_equal_seg.pkl'))
        pimgdir = '../pixpred'
        '''
        dirlist = os.listdir(pimgdir)

        WORDBB = []
        for filename in dirlist:
                if not filename.endswith('.png'):
                        continue
                print filename
                predimg = image_load(filename, pimgdir)
                bblist1 = bimg2bblist(predimg)
                if len(bblist1) == 0:
                        continue
                relationMat = bblist2relationMat(bblist1, classifier)
                wordbb = relationMat2wordbb(bblist1, relationMat)
                wordbb = wordbbMerge(wordbb)
                WORDBB.append([filename, wordbb])
                #plotbb(predimg, bblist1, wordbb)
        return WORDBB

def cwCombine(wordDataList, charDataList):
        '''cwCombine(wordDataList, charDataList)

        Combine character and word bounding box list
        Input:
        wordDataList: list, containing word bounding box, 
        charDataList: list, containing character bounding box,
        Return:
        list, containing image name string, word bounding box list and character bounding box list
        '''
        data = []
        for imgname0, bblist0 in wordDataList:
                for imgname1, bblist1 in charDataList:
                        if imgname0 == imgname1:
                                data.append((imgname0, bblist0, bblist1))
        return data

def isSameWord(p1, p2, bblist):
        ''' isSameWord(p1, p2, bblist)

        check if two center points are in the same word,
        Input:
        p1: array like, center point coordinate,
        p2: array like, center point coordinate,
        bblist: word bounding box list
        Return:
        Boolean, True if two center points are in the same word '''
        for bb in bblist:
                x, y, w, h = bb
                if p1[0] > x and p1[0] < x+w and p1[1] > y and p1[1] < y+h:
                        if p2[0] > x and p2[0] < x+w and p2[1] > y and p2[1] < y+h:
                                return True
                        else:
                                return False

def plotbb_train(DataList):
        train_feature = []
        train_label = []
        for imgname, bblist0, bblist1 in DataList:
                print imgname
                img = image_load(imgname, imdir)

                fig = plt.figure(frameon = False)
                fig.set_size_inches(10, 10)
                ax = plt.Axes(fig, [0., 0., 1., 1.], )
                ax.set_axis_off()
                fig.add_axes(ax)

                ax.imshow(img.transpose(), cmap = plt.get_cmap('gray'))
                for bb in bblist0:
                        x, y, w, h = bb
                        x1 = numpy.arange(x, x+w)
                        y1 = numpy.ones(len(x1)) * y
                        x2 = numpy.arange(x, x+w)
                        y2 = numpy.ones(len(x2)) * (y+h)

                        y3 = numpy.arange(y, y+h)
                        x3 = numpy.ones(len(y3)) * x
                        y4 = numpy.arange(y, y+h)
                        x4 = numpy.ones(len(y4)) * (x+w)

                        plt.plot(x1, y1, 'b')
                        plt.plot(x2, y2, 'b')
                        plt.plot(x3, y3, 'b')
                        plt.plot(x4, y4, 'b')

                for bb in bblist1:
                        x, y, w, h = bb
                        x1 = numpy.arange(x, x+w)
                        y1 = numpy.ones(len(x1)) * y
                        x2 = numpy.arange(x, x+w)
                        y2 = numpy.ones(len(x2)) * (y+h)

                        y3 = numpy.arange(y, y+h)
                        x3 = numpy.ones(len(y3)) * x
                        y4 = numpy.arange(y, y+h)
                        x4 = numpy.ones(len(y4)) * (x+w)

                        plt.plot(x1, y1, 'g')
                        plt.plot(x2, y2, 'g')
                        plt.plot(x3, y3, 'g')
                        plt.plot(x4, y4, 'g')

                centerList = []
                for bb in bblist1:
                        x, y, w, h = bb
                        x0 = x+w/2
                        y0 = y+h/2
                        centerList.append([x0, y0])
                        plt.scatter([x0], [y0], c = 'red')

                clength = len(centerList)
                for i in range(clength):
                        for j in range(clength/2):
                                plt.plot([centerList[i][0], centerList[j][0]], [centerList[i][1], centerList[j][1]], 'y', alpha = 0.5)

                centerList = numpy.asarray(centerList)
                distMat = squareform(pdist(centerList))
                if distMat.shape[0] <= 1:
                        continue

                edgeMat = minimum_spanning_tree(distMat)
                for edge in edgeMat:
                        i, j = edge
                        p1 = centerList[i, :]
                        p2 = centerList[j, :]
                        train_feature.append([bblist1[i], bblist1[j]])
                        if isSameWord(p1, p2, bblist0):
                                train_label.append(1)
                                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b', lw = 2)
                        else:
                                train_label.append(0)
                                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r', lw = 2)
                #plt.tight_layout()
                plt.show()
                #raw_input()
        return train_feature, train_label

def wordGraph_train():
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC as svm

        from sklearn.metrics import f1_score, precision_score, recall_score
        from sklearn.cross_validation import train_test_split
        import pickle

        imdir = '../../icdar2013/task21_22/train/image/'
        worddir = '../../icdar2013/task21_22/train/word_label/'
        chardir = '../../icdar2013/task21_22/train/char_label/'

        mywordparser = parseWord2013()
        mycharparser = parseChar2013()
        wordDataList = mywordparser.parseData(imdir, worddir)
        charDataList = mycharparser.parseData(imdir, chardir)
        DataList = cwCombine(wordDataList, charDataList)

        train_feature, train_label = plotbb_train(DataList)
        train_feature = numpy.asarray(train_feature)
        train_label = numpy.asarray(train_label)
        numpy.save('train_feature_seg', train_feature, )
        numpy.save('train_label_seg', train_label, )

        train_feature_width = train_feature[:, :, 2]
        train_feature_height = train_feature[:, :, 3]
        train_feature_area = train_feature[:, :, 2] * train_feature[:, :, 3]
        train_feature_aspectRatio = numpy.float32(train_feature_width) / numpy.float32(train_feature_height)
        train_feature_x = train_feature[:, :, 0]
        train_feature_y = train_feature[:, :, 1]
        train_feature_cx = train_feature_x+train_feature_width/2
        train_feature_cy = train_feature_y+train_feature_height/2
        edge_width_dif = abs(train_feature_width[:, 0] - train_feature_width[:, 1])
        edge_height_dif = abs(train_feature_height[:, 0] - train_feature_height[:, 1])
        edge_area_dif = abs(train_feature_area[:, 0] - train_feature_area[:, 1])
        edge_aspectRatio_dif = abs(train_feature_aspectRatio[:, 0] - train_feature_aspectRatio[:, 1])
        edge_cx_dis = abs(train_feature_cx[:, 0] - train_feature_cx[:, 1])
        edge_cy_dis = abs(train_feature_cy[:, 0] - train_feature_cy[:, 1])
        edge_width_mean = abs(train_feature_width[:, 0] + train_feature_width[:, 1]) / 2
        edge_height_mean = abs(train_feature_height[:, 0] + train_feature_height[:, 1]) / 2
        edge_area_mean = abs(train_feature_area[:, 0] + train_feature_area[:, 1]) / 2
        edge_aspectRatio_mean = abs(train_feature_aspectRatio[:, 0] + train_feature_aspectRatio[:, 1]) / 2
        edge_cu_dis = numpy.power(numpy.power(edge_cx_dis,2)+ numpy.power(edge_cy_dis, 2),0.5) # center Euclidean distance
        edge_bd_dis = numpy.maximum(edge_cx_dis - edge_width_mean, edge_cy_dis - edge_height_mean)# closest boundary distance
        edge_cu_dis_norm = numpy.float32(edge_cu_dis) / numpy.float32(edge_area_mean)
        edge_bd_dis_norm = numpy.float32(edge_bd_dis) / numpy.float32(edge_area_mean)
        edge_width_dif_norm = numpy.float32(edge_width_dif) / numpy.float32(edge_width_mean)
        edge_height_dif_norm = numpy.float32(edge_height_dif) / numpy.float32(edge_height_mean)
        edge_cx_dis_norm = numpy.float32(edge_cx_dis) / numpy.float32(edge_width_mean)
        edge_cy_dis_norm = numpy.float32(edge_cy_dis) / numpy.float32(edge_height_mean)
        edge_angle_dis = numpy.arctan(numpy.float32(edge_cx_dis) / numpy.float32(edge_cy_dis))

        feature = numpy.asarray([edge_cu_dis, edge_cu_dis_norm, edge_bd_dis, edge_bd_dis_norm, edge_cx_dis, edge_cx_dis_norm,
                edge_cy_dis, edge_cy_dis_norm, edge_width_dif, edge_width_dif_norm, 
                edge_height_dif, edge_height_dif_norm, edge_angle_dis]).T
        # replace NaN or Inf with average values ###########
        feature_mean = numpy.mean(feature, axis = 0)
        for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                        if numpy.isnan(feature[i, j]) or numpy.isinf(feature[i, j]):
                                print 'NaN(Inf) found!'
                                feature[i, j] = feature_mean[j]

        print 'w/o sample equalization'
        f_train, f_test, l_train, l_test = train_test_split(feature, train_label, test_size = 0.2)

        classifier = GradientBoostingClassifier(max_depth = 1)
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('adaboost_unequal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'AdaBoost classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'


        classifier = svm(kernel = 'linear')
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('svm_unequal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'SVM classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        classifier = RandomForestClassifier()
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('randomForest_unequal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'Random Forest classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        print 'Equalized Samples: SMOTE algorithm'
        feature0 = feature[train_label == 0, ...]
        label0 = train_label[train_label == 0]

        feature1 = feature[train_label == 1, ...]
        label1 = train_label[train_label == 1]

        feature0 = SMOTE(feature0, len(feature1)/len(feature0)*100, 3)
        label0 = numpy.zeros(len(feature0))

        f_train = numpy.concatenate([feature0, feature1])
        l_train = numpy.concatenate([label0, label1])

        classifier = GradientBoostingClassifier(max_depth = 1)
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('adaboost_smote_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'AdaBoost classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        classifier = svm(kernel = 'linear')
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('svm_smote_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'SVM classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        classifier = RandomForestClassifier()
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('randomForest_smote_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'Random Forest classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        print 'Sample equalization with random selection'
        feature0 = feature[train_label == 0, ...]
        label0 = train_label[train_label == 0]

        feature1 = feature[train_label == 1, ...]
        label1 = train_label[train_label == 1]

        idx = numpy.random.choice(feature1.shape[0], feature0.shape[0], replace = False)
        feature1 = feature1[idx, ...]
        label1 = label1[idx, ...]

        f_train = numpy.concatenate([feature0, feature1])
        l_train = numpy.concatenate([label0, label1])

        classifier = GradientBoostingClassifier(max_depth = 1)
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('adaboost_equal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'AdaBoost classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        classifier = svm(kernel = 'linear')
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('svm_equal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'SVM classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'

        classifier = RandomForestClassifier()
        classifier = classifier.fit(f_train, l_train)
        pickle.dump(classifier, open('randomForest_equal_seg.pkl', 'w'))

        pred_train = classifier.predict(f_train)
        pred_test = classifier.predict(f_test)

        f1_train = f1_score(l_train, pred_train)
        precision_train = precision_score(l_train, pred_train)
        recall_train = recall_score(l_train, pred_train)

        f1_test = f1_score(l_test, pred_test)
        precision_test = precision_score(l_test, pred_test)
        recall_test = recall_score(l_test, pred_test)

        print 'Random Forest classifier training(testing):'
        print 'precision', '%.4f' % precision_train, '(', '%.4f' % precision_test, ')'
        print 'recall', '%.4f' % recall_train, '(', '%.4f' % recall_test, ')'
        print 'f1 score', '%.4f' % f1_train, '(', '%.4f' % f1_test, ')'
        print '\r\n'
                
        return

