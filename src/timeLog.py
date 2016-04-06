#!/usr/bin/env python
#title           :timeLog.py
#description     :log files recording computation time.
#author          :siyu zhu
#date            :June 30rd, 2014
#version         :0.1
#usage           :from timeLog import timeLog
#notes           :
#python_version  :2.7
from fileOp import make_dir
import time
import os

class timeLog:

    def __init__(self, filename):
        make_dir(os.path.dirname(filename))
        with open(filename, 'w') as f:
                f.write('')
        self.gtime = time.time()
        self.filename = filename
        self.ninstance = 0
        self.start_ = 0
        self.message = []
        return

    def start(self, message = ''):
        if self.start_ == 1:
            print 'timer already started...'
            return
        self.start_ = 1
        self.time = time.time()
        self.message.append(message)
        self.ninstance += 1
        return

    def end(self, message = ''):
        if self.start_ == 0:
            print 'timer is not started yet ...' 
            return
        self.start_ = 0
        etime = str(time.time() - self.time)
        self.message.append(message)
        self.message.append(' Elapse Time: '+etime+'\r\n')
        mymessage = ' '.join(self.message)
        with open(self.filename, 'a') as f:
            f.write(mymessage)
        self.message = []
        del self.time
        return

    def final(self, message = ''):
        if self.ninstance == 0:
            print 'no timer founded ...'
            return
        etime = str(time.time() - self.gtime) 
        atime = str(float(etime) / self.ninstance)
        self.message.append(message)
        self.message.append(' Total Elapse Time: '+etime+'\r\n')
        self.message.append(' Average Elapse Time: '+atime+'\r\n')
        mymessage = ' '.join(self.message)
        with open(self.filename, 'a') as f:
            f.write(mymessage)
        return 

