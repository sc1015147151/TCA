# encoding=utf-8

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from scipy import sparse
import math

'''
print(dir(svm))
'''
class TCA:
    #dim=5
    kerneltype = 'rbf0'
    kernelparam = 0.1


    def __init__(self, dim=6, kerneltype='rbf0', kernelparam=0.1, mu=0.03,d=0):
        '''
        Init function
        :param dim: dims after tca (dim <= d)
        :param kerneltype: 'rbf' | 'linear' | 'poly' (default is 'rbf')
        :param kernelparam: kernel param
        :param mu: param
        '''
        self.dim = dim
        self.kernelparam = kernelparam
        self.kerneltype = kerneltype
        self.mu = mu

    def get_L(self, n_src, n_tar,d):
        '''
        Get index matrix
        :param n_src: num of source domain 
        :param n_tar: num of target domain
        :return: index matrix L
        '''
        '''

        return L
        '''
        if d==0:
            L1=(1./ n_src)*np.identity(n_src)
            L2=(-1. / n_src)*np.identity(n_src)
            L_up=np.hstack((L1, L2))
            L_down=np.hstack((L2, L1))
            L=np.vstack((L_up, L_down))
            return L
        else :
            L_ss = (1. / (n_src * n_src)) * np.full((n_src, n_src), 1)
            L_st = (-1. / (n_src * n_tar)) * np.full((n_src, n_tar), 1)
            L_ts = (-1. / (n_tar * n_src)) * np.full((n_tar, n_src), 1)
            L_tt = (1. / (n_tar * n_tar)) * np.full((n_tar, n_tar), 1)
            L_up = np.hstack((L_ss, L_st))
            L_down = np.hstack((L_ts, L_tt))
            L = np.vstack((L_up, L_down))
        
        
    
    def get_kernel(self, kerneltype, kernelparam, x1, x2=None):
        '''
        Calculate kernel for TCA (inline func)
        :param kerneltype: 'rbf' | 'linear' | 'poly'
        :param kernelparam: param
        :param x1: x1 matrix (n1,d)
        :param x2: x2 matrix (n2,d)
        :return: Kernel K
        '''
        n1, dim = x1.shape
        K = None
        if x2 is not None:
            n2 = x2.shape[0]
        if kerneltype == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif kerneltype == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1, x2.T), kernelparam)
            else:
                K = np.power(np.dot(x1, x1.T), kernelparam)
        elif kerneltype == 'rbf0':
            if x2 is not None:
                sum_x2 = np.sum(np.multiply(x2, x2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (
                    np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) + np.tile(sum_x2, (1, n1)) - 2 * np.dot(x2,
                                                                                                                    x1.T)) / (
                               dim * 2 * kernelparam))
            else:
                P = np.sum(np.multiply(x1, x1), axis=1)
                P = P.reshape((len(P), 1))
                K = np.exp(
                    -1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * kernelparam))
        # more kernels can be added
        return K

    def fit_transform(self, x_src, x_tar, x_tar_o,d):
        '''
        TCA main method. Wrapped from Sinno J. Pan and Qiang Yang's "Domain adaptation via transfer component ayalysis. IEEE TNN 2011" 
        :param x_src: Source domain data feature matrix. Shape is (n_src,d)
        :param x_tar: Target domain data feature matrix. Shape is (n_tar,d)
        :param x_tar_o: Out-of-sample target data feature matrix. Shape is (n_tar_o,d)
        :return: tranformed x_src_tca,x_tar_tca,x_tar_o_tca
        '''
        n_src = x_src.shape[0]
        n_tar = x_tar.shape[0]
        n_tar_o = x_tar_o.shape[0]
        X = np.vstack((x_src, x_tar))
        L = self.get_L(n_src, n_tar,d=0)
        L[np.isnan(L)] = 0
        K = self.get_kernel(self.kerneltype, self.kernelparam, X)
        K[np.isnan(K)] = 0
        if x_tar_o is not None:
            K_tar_o = self.get_kernel(self.kerneltype, self.kernelparam, X, x_tar_o)

        H = np.identity(n_src + n_tar) - 1. / (n_src + n_tar) * np.ones(shape=(n_src + n_tar, 1)) * np.ones(
            shape=(n_src + n_tar, 1)).T
        forPinv = self.mu * np.identity(n_src + n_tar) + np.dot(np.dot(K, L), K)
        forPinv[np.isnan(forPinv)] = 0
        Kc = np.dot(np.dot(np.dot(np.linalg.pinv(forPinv), K), H), K)
        Kc[np.isnan(Kc)] = 0

        D, V = np.linalg.eig(Kc)
        eig_values = D.reshape(len(D), 1)
        eig_values_sorted = np.sort(eig_values[::-1], axis=0)
        index_sorted = np.argsort(-eig_values, axis=0)
        V = V[:, index_sorted]
        V = V.reshape((V.shape[0], V.shape[1]))
        x_src_tca = np.dot(K[:n_src, :], V)
        x_tar_tca = np.dot(K[n_src:, :], V)
        if x_tar_o is not None:
            x_tar_o_tca = np.dot(K_tar_o, V)
        else:
            x_tar_o_tca = None
        x_src_tca = np.asarray(x_src_tca[:, :self.dim], dtype=float)
        x_tar_tca = np.asarray(x_tar_tca[:, :self.dim], dtype=float)
        if x_tar_o is not None:
            x_tar_o_tca = x_tar_o_tca[:, :self.dim]
        return V,x_src_tca, x_tar_tca, x_tar_o_tca
    def classify_svm(self,x_src_tca, y_src,x_tar_o_tca,y_tar_o):
        clf=svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovr')
        clf.fit(x_src_tca,y_src)
        #y_tar_o_pre = svm.SVC.predict(x_tar_o_tca)
        y_tar_o_pre = clf.predict(x_tar_o_tca)
        acc_tar_o=metrics.accuracy_score(y_tar_o, y_tar_o_pre)
        kappa_tar=metrics.cohen_kappa_score(y_tar_o, y_tar_o_pre)
        return acc_tar_o, y_tar_o_pre, kappa_tar
    def fit(self,left_part,right_part):
        self.src_x=
        self.src_y=
        self.tar_x=
        self.tar_y=
    def predict(self,left_part,right_part):
        #o_src_x_TCA=
        #o_src_y=
        o_tar_x_TCA=
        o_tar_y=
        V,src_x_TCA, tar_x_TCA, o_tar_x_TCA=fit_transform(self, self.src_x, self.tar_x, o_tar_x,d)
        clf=svm.SVC(C=1,kernel='rbf',gamma=10,decision_function_shape='ovr')
        clf.fit(x_src_TCA,self.src_y)
        #y_tar_o_pre = svm.SVC.predict(x_tar_o_tca)
        o_tar_y_PRE = clf.predict(o_tar_x_TCA)
        self.acc=metrics.accuracy_score(o_tar_y, o_tar_y_PRE)
        return o_tar_y_PRE
    def score():
        return self.acc