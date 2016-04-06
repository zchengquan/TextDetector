#!/usr/bin/env python
#title           :zipFile.py
#description     :convert files in a list into single zip file.
#author          :siyu zhu
#date            :Dec 16th, 2014
#version         :0.1
#usage           :
#notes           :
#python_version  :2.7
#==============================================================================
# import modules
import zipfile 
import os
from sys import argv 
#==============================================================================

def zipFile(zipfilename = '../pred.zip', sourcedir = '../bbtxt/icdar2013wordadaboost'):
	try:
		filelist = os.listdir(sourcedir)
		zf = zipfile.ZipFile(zipfilename, 'w')
		for filename in filelist:
			zf.write(os.path.join(sourcedir, filename), filename)
		zf.close()
		print zipfilename, 'created.'
	except:
		print 'format error: zipFile(target_file_name, source_dir_name)'
	
def main(arg):
	zipFile(arg[1], arg[2])	

if __name__ == '__main__':
	main(argv)
