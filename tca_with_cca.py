# -*- coding: cp936 -*-
def tca_with_cca(k=1):
    from sklearn.cross_validation import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn import tree
    from sklearn import metrics
    import pandas as pd
    from cca import cca
    import numpy as np
    import tca 
    import ascr
    file_path1='sample.csv'
    file_path2='out_of_sample.csv'
#train_sample = pd.read_csv(file_path1)
    train_sample=pd.read_csv(file_path1).sample(frac=0.05,replace=True, random_state=k, axis=0)
    #train_sample=ascr.ascr(ascr.ascr(ascr.ascr(ascr.ascr(ascr.ascr(ascr.ascr(pd.read_csv(file_path1).sample(frac=0.02,replace=False, random_state=11, axis=0),'S1','T1'),'S2','T2'),'S3','T3'),'S4','T4'),'S5','T5'),'S6','T6')

    cca_s,cca_t=cca(H1=train_sample.loc[:,'S1':'S_S7'],H2=train_sample.loc[:,'T1':'T_S7'])
##################################################################    

    x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(
        np.append(cca_s,cca_t,axis = 1) ,
        pd.concat([train_sample[['S_Y']], train_sample[['T_Y']]], axis=1, join='inner'),
        test_size=0.25, 
        random_state=k)
##################################################################
    #x_test_st=x_test_st.values

    y_test_st=y_test_st.values

    y_train_st=y_train_st.values
    #x_train_st=x_train_st.values    

    x_src = x_train_st[:, 0:6]

    x_tar = x_train_st[:, 6:12]

    y_src=y_train_st[:, 0:1]

    y_tar=y_train_st[:, 1:2]

    x_tar_o= x_test_st[:, 6:12]

    y_tar_o=y_test_st[:, 1:2]


    my_tca = tca.TCA(dim=6)

    V,x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(x_src, x_tar,x_tar_o) 

    acc_tar_o,y_tar_o_pre,kappa_tar = my_tca.classify_svm(x_src_tca, y_src, x_tar_o_tca, y_tar_o)

    #print('SVM',acc_tar_o,kappa_tar)
#

    dr = tree.DecisionTreeClassifier()

    dr = dr.fit(x_src_tca, y_src)

    print('DT',metrics.accuracy_score(y_tar_o, dr.predict(x_tar_o_tca)),
          metrics.cohen_kappa_score(y_tar_o,dr.predict(x_tar_o_tca)))
    
    temp=pd.DataFrame(data={'SVM': [acc_tar_o], 'DT': [metrics.accuracy_score(y_tar_o,dr.predict(x_tar_o_tca))]})
    print(temp) 
 
    for i in ('sgd','adam','lbfgs'):
        nn = MLPClassifier(solver=i, alpha=1e-5,hidden_layer_sizes=(84, 6), random_state=k+31)
        nn = nn.fit(x_src_tca, y_src)
        temp=  pd.concat([temp, pd.DataFrame(data={i:[metrics.accuracy_score(y_tar_o, nn.predict(x_tar_o_tca))]})], axis=1, join='inner')
        print(temp)

        print(i,'NN',metrics.accuracy_score(y_tar_o, nn.predict(x_tar_o_tca)),
              metrics.cohen_kappa_score(y_tar_o, nn.predict(x_tar_o_tca))) 
   
    record=pd.read_csv('record.csv')
    print(record)
    record=pd.concat([record,temp], axis=0, join='inner')
    record.to_csv('record.csv',index=False) 
    print(record)

if __name__=='__main__':  
   for i in range (1,1000):
       tca_with_cca.tca_with_cca(i)
