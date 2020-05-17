import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn import svm
from sklearn import metrics
from scipy import sparse
import math
def Cca(H1 =[[1,0],[0,2],[3,0],[0,4]],H2 =[[1,0],[0,2],[3,0],[0,4]],dim=2,rcov1=0,rcov2=0):  
    #H1 =np.diag(np.arange(1,5,1) )
    #H2 =np.diag(np.arange(3,7,1) )
    N=np.size(H1,0)
    d1=np.size(H1,1)
    d2=np.size(H2,1)
    #dim=4
    mean_H1=np.mean(H1,0)
    mean_H2=np.mean(H2,0)
    #rcov1=0
    #rcov2=0
    #print('需要相似计算的两个矩阵\n H1 \n',H1,'\n H2 \n',H2)
    h1=H1-np.tile(mean_H1, (N,1))
    h2=H2-np.tile(mean_H2, (N,1))
    S11 = np.dot(h1.T,h1)/(N-1)+rcov1*np.ones(d1)
    S22 = np.dot(h2.T,h2)/(N-1)+rcov2*np.ones(d2)
    S12 = np.dot(h1.T,h2)/(N-1)
    w, v = LA.eig(np.diag((1, 2, 3)))
    D1,V1=LA.eig(S11)
    D2,V2=LA.eig(S22)
    #indexD2= [i for i in range(len(D2)) if D2[i]>0.0000001]
    #indexD1= [i for i in range(len(D1)) if D1[i]>0.0000001]
    #D1 = D1[indexD1]
    #V1 = V1[:,indexD1]
    #D2 = D2[indexD2]
    #V2 = V2[:,indexD2]
    #np.diag(np.power(D1,-0.5))
    K11=np.dot(np.dot(V1,np.diag(np.power(D1,-0.5))),V1.T)
    K22=np.dot(np.dot(V2,np.diag(np.power(D2,-0.5))),V2.T)
    T=np.dot(np.dot(K11,S12),K22)
    u,s,vh=LA.svd(T)
    #print(T)
    s=np.diag(s)
    #print (np.dot(np.dot(u,s),vh))

    A = np.dot(K11,u[:,0:dim])
    B = np.dot(K22,vh.T[:,0:dim])
    #print ('u \n',u,'\n s \n',s,'\n vh \n',vh.T)
    #print('矩阵A\n',A)
    #print('矩阵B\n',B)
    #print(np.dot(H1,A),'\n')
    #print(np.dot(H2,B),'\n')
    return np.dot(H1,A),np.dot(H2,B)
if __name__=='__main__':  
    cca()  