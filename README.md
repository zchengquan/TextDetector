DPRL Natural Scene Text Detector
---------------

DPRL Natural Scene Text Detector
Copyright (c) 2012-2016 Siyu Zhu, Richard Zanibbi

DPRL Natural Scene Text Detector is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

DPRL Natural Scene Text Detector is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with DPRL Natural Scene Text Detector.  If not, see <http://www.gnu.org/licenses/>.

Contact:
	- Siyu Zhu: sxz8564@rit.edu
	- Richard Zanibbi: rlaz@cs.rit.edu 

## Documentation
* * *

This document is about the Text Detection in Natural Scene in the paper [A Text Detection System for Natural Scenes with Convolutional Feature Learning and Cascaded Classification].
The details about feature learning, detector training and testing, region growing and segmentation can be found in the paper.
This file should serve as a starting point for developers who are interested in understanding the underlying operations that occur within DPRL Natural Scene Text Detector. Each file associated with Text Conv Natural Scene Text Detector has a brief and concise overview of major operations of the file and how it is used within the system. 

This is a software that finds text in natural scenes using a variety of cues. Our novel data-driven method incorporates coarse-to-fine detection of character pixels using convolutional features (Text-Conv), followed by extracting connected components (CCs) from characters using edge and color features, and finally performing a graph-based segmentation of CCs into words (Word-Graph). For Text-Conv, the initial detection is based on convolutional feature maps similar to those used in Convolutional Neural Networks (CNNs), but learned using Convolutional k-means. Convolution masks defined by local and neighboring patch features are used to improve detection accuracy. The Word-Graph algorithm uses contextual information to both improve word segmentation and prune false character/word detections. Different definitions for foreground (text) regions are used to train the detection stages, some based on bounding box intersection, and others on bounding box and pixel intersection. Our system obtains pixel, character, and word detection f-measures of 93.14\%, 90.26\%, and 86.77\% respectively for the ICDAR 2015 Robust Reading Focused Scene Text dataset, out-performing state-of-the-art systems. 

## Installation / Download
* * *
Text-Conv is text detection software that requires no installation. Simply download this software along with training and testing data.

The source code is included in `/src`.  The final results are `/src/pred.zip`.


Directory `/data/myexpe` contains the input data, intermedia and final results of text detection. The images of testing data are in `/data/myexpe/raw`; The patch features are in `/data/myexpe/npy`; The patch detection results are in `/data/myexpe/pre`; The pixel level maps are in `/data/myexpe/roitest`; The bounding box text files are in `/data/myexpe/pretxt`.

Directory `/codebooks` contains codebook trained by convolutional k-means and used for text detection.

Source files and trained classifiers can be downloaded from the software tab of the lab website or via GitHub. To train the system, one also need to download data from [ICDAR robust reading competition website], including training images and bounding box text files. Our system is compatible with both ICDAR 2003 and ICDAR 2013(2015) training and testing data. Due to the size of the data, they are not included in our packege. 

## Usage
* * *

* learning features with convolutional k-means:

		cd src/cluster
		python use_ckmean.py

* training detector:

		cd src/
		python textTrain.py
		
* example of testing detector:

		cd src/
		python textDet.py

***
`textDet.py` performs text detection.

`textTrain.py` performs text detector training.

`detector.py` contains class for convolutional patch based detector.

`wordGraph.py` contains class for word graph segmentation training and testing. 

`parseData.py` contains class for parse input ground truth and images from ICDAR dataset.

***
`arrayOp.py` contains functions for numpy array operations.

`bbOp.py` contains functions for bounding box operations.

`fileOp` contains functions for file operation.

`imgOp` contains functions for image operations.

***
`classifier.py` contain the classifier class.

`featExt` contain function to extract convolution features.

`featLearn` contains function for feature learning.

`imageProcess` contains functions for image pre-processing.

`mail.py` contains function for sending emails for notification.

`para_classifier.py` contains example kernel function for training a classifier using parallel process.

`plot.py` contains functions for plotting performance.

`timeLog.py` contains class for elapse time counting.

`zipFile.py` contains function for compress files.

***
Directory `/src/cluster` contains function for convolutional k-means training.

Directory `/src/classifiers` contain classifiers for cascaded classification.

***

The system is composed by multistages of cascaded filtering. The first stage classifies the patches extracted by `detector.py` and produces numpy array containing 30 differnet scales of detection results. The second stage performs fine detection using `detector.py` upon the first stage results with small steps. Region growing is performed by function `regiongrow` to form connected components. The connect comoponents are then segmented using `wordGraph.py`.


##Dependencies
* * *

Packages required for python packages: scipy, numpy, scikit-learn, scikit-image, theano, matplotlib, cython, joblib.

* The package _scipy_ and _numpy_ are used for basic matrix operation and computation, which can be found at: 
	[https://www.scipy.org] and
	[http://www.numpy.org]
* The package _scikit-learn_ provides basic machine learning algorithms, and can be found at:[http://scikit-learn.org/stable/]
* The package _scikit-image_ provides basic image transformation, and can be found at: [http://scikit-image.org]
* The package _theano_ provides basic function that allowing GPU computation, and can be found at:[http://deeplearning.net/software/theano/]
* The package _matplotlib_ provides basic function allowing plot and drawing figures: [http://matplotlib.org]
* The package _cython_ provides acceleration for python code by converting it to c code: [http://cython.org]
* The package _joblib_ provides the accleration for python by using multiple cores parallel process:[https://pypi.python.org/pypi/joblib]

[https://www.scipy.org]:https://www.scipy.org
[http://www.numpy.org]:http://www.numpy.org
[http://scikit-learn.org/stable/]: http://scikit-learn.org/stable/
[http://scikit-image.org]: http://scikit-image.org
[http://deeplearning.net/software/theano/]: http://deeplearning.net/software/theano/
[http://matplotlib.org]: http://matplotlib.org
[http://cython.org]:http://cython.org
[https://pypi.python.org/pypi/joblib]:https://pypi.python.org/pypi/joblib


* * *

*Visit the DPRL page for more information about the lab and projects.*

*This material is based upon work supported by the National Science Foundation (USA) under Grant Nos. IIS-1016815 and HCC-1218801.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) 
and do not necessarily reflect the views of the National Science Foundation.*
