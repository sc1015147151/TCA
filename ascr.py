import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn import svm
from sklearn import metrics
from scipy import sparse
import math
import matplotlib.pyplot as plt
def ascr(train_sample,sBandNum,tBandNum,percentile=95):
  same_train_sample=train_sample[train_sample['S_Y'] ==train_sample['T_Y']  ]
  water_same_train_sample=same_train_sample[(same_train_sample['S_Y']==2)]
  land_same_train_sample=same_train_sample[(same_train_sample['S_Y']==1)|(same_train_sample['S_Y']==3)|(same_train_sample['S_Y']==5|(same_train_sample['S_Y']==4))]

  land_same_train_sample=same_train_sample[(same_train_sample['S_Y']==1)|(same_train_sample['S_Y']==3)|(same_train_sample['S_Y']==5)]
  water_same_train_sample=same_train_sample[(same_train_sample['S_Y']==2)|(same_train_sample['S_Y']==4)]

  land_mean=land_same_train_sample.mean(0)

  water_mean=water_same_train_sample.mean(0)
  plt.scatter(land_same_train_sample[sBandNum],land_same_train_sample[tBandNum],s=1)
  plt.scatter(water_same_train_sample[sBandNum],water_same_train_sample[tBandNum],s=1)



  k=(water_mean[tBandNum]-land_mean[tBandNum])/(water_mean[sBandNum]-land_mean[sBandNum])
  b=water_mean[tBandNum]-k*water_mean[sBandNum]                              

  #plt.scatter(land_same_train_sample[sBandNum],land_same_train_sample[tBandNum],s=1)
  #plt.scatter(water_same_train_sample[sBandNum],water_same_train_sample[tBandNum],s=1,c='g')
  #plt.plot([water_mean[sBandNum]*0.5,water_mean[tBandNum]*2],[land_mean[sBandNum]*0.5,land_mean[tBandNum]*2])
  #plt.show()

  d1=[]
  for indexs in  same_train_sample.index: 
      d1.append(abs(k * same_train_sample.loc[indexs][sBandNum] - same_train_sample.loc[indexs][tBandNum] + b)/((-1)*(-1) + k * k)**0.5)
      #print()

 # plt.hist(d1, bins=1024, facecolor='green', alpha=0.5)  
 # plt.show()
#print(len(d1),d1)
  d2=[]
  for indexs in  same_train_sample.index: 
    
    if abs((k * same_train_sample.loc[indexs][sBandNum] - same_train_sample.loc[indexs][tBandNum] + b)/((-1)*(-1) + k * k)**0.5)>(np.percentile(d1,percentile)):
               
        same_train_sample=same_train_sample.drop(indexs,axis=0)
    else :
        d2.append(abs(k * same_train_sample.loc[indexs][sBandNum] - same_train_sample.loc[indexs][tBandNum] + b)/((-1)*(-1) + k * k)**0.5)      
  #plt.hist(d2, bins=1024, facecolor='green', alpha=0.5)  
  #plt.show()
  #same_train_sample.info()
  return same_train_sample
