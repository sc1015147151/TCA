# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:00:56 2018

@author: xin
"""
'''
#from sklearn import preprocessing
#from sklearn.pipeline import make_pipline
import numpy as np
from sklearn import cross_validation
from sklearn import svm

def estimator(x_src,x_tar,y_src,y_tar):
    clf = svm.SVC(kernel='rbf', C=5).fit(x_src,y_src)
    #score=model_selction.cross_val_score(clf,x_tar_tranformed,y_tar,cv=5)
    clf_score1=clf.score(x_tar,y_tar)
    clf_score2=clf.score(x_tar_o,y_tar_o)
    return clf_score1,clf_score2
'''
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

def estimator(x_src,x_tar,y_src,y_tar):
     data=np.vstack((x_src,x_tar))
     label=np.vstack((y_src,y_tar))
     clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='rbf',C=1,gamma=20))
     score=cross_val_score(clf,data,label,cv=2)
     return score
    
if __name__ == '__main__':
    file_path='norm_src_tar.csv'
    data = np.loadtxt(file_path, delimiter=',')
    x_src = data[:6000, 0:1]
    x_tar = data[:6000, 7:8]
    y_src=data[:6000, 6:7]
    y_tar=data[:6000, 13:14]
    x_tar_o=data[6000:12000, 0:6]
    y_tar_o=data[6000:12000, 6:7]
    
    print(estimator(x_src,x_tar,y_src,y_tar))