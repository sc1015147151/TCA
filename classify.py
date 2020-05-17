# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:11:57 2018

@author: xin
"""

import numpy as np
from sklearn import svm
from sklearn import metrics
from scipy import sparse
import math

def classify_svm(x_src, y_src,x_tar_o,y_tar_o):
       # sum=0
       #n_tar_o=x_tar_o.shape[0]
        clf=svm.SVC(C=1,kernel='rbf',gamma=0.1,decision_function_shape='ovr')
        clf.fit(x_src,y_src)
        #y_tar_o_pre = svm.SVC.predict(x_tar_o_tca)
        y_tar_pre = clf.predict(x_tar_o)
        acc_tar_o=metrics.accuracy_score(y_tar_o, y_tar_pre)
        kappa_tar=metrics.cohen_kappa_score(y_tar_o,y_tar_pre)
        return acc_tar_o, y_tar_pre,kappa_tar
    
if __name__ == '__main__':
    file_path1='train_sample.csv'
    file_path2='out_of_sample.csv'
    train_sample = np.loadtxt(file_path1, delimiter=',')
    out_of_sample = np.loadtxt(file_path2, delimiter=',')
    x_src = train_sample[:5000, 0:6]
    x_tar = train_sample[:5000, 7:13]
    y_src=train_sample[:5000, 6:7]
    y_tar=train_sample[:5000, 13:14]
    x_tar_o=out_of_sample[:5000, 0:6]
    y_tar_o=out_of_sample[:5000, 6:7]

     #example usage
    
    #print(acc_tar_o)
    print(classify_svm(x_src, y_src, x_tar_o, y_tar_o))
    acc_tar_o,y_tar_pre,kappa_tar=classify_svm(x_src, y_src, x_tar_o, y_tar_o)
    np.savetxt('y_tar_pre.csv', y_tar_pre, delimiter=',', fmt='%.6f')
