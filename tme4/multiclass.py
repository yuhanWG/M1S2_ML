from arftools import *
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib import cm

N=10
def one_versus_one(datax,datay,testx,testy):
    svms=[]
    y=np.unique(datay)
    y1_=[]
    y2_=[]
    for y1 in y:
        for y2 in y:
            if (y1<y2):
                train_x,train_y=datax[(datay==y1)|(datay==y2)],datay[(datay==y1)|(datay==y2)]
                #test_x,test_y=testx[(testy==y1)|(testy==y2)],testy[(testy==y1)|(testy==y2)]
                y1_.append(y1)
                y2_.append(y2)
                svms.append(svm.SVC(kernel='linear').fit(train_x,train_y))
    return svms


def one_versus_all(datax,datay,testx,testy):
    svms=[]
    for i in range(len(np.unique(datay))):
        data_y=np.where(datay==i,-1,1)
        svms.append(svm.SVC(kernel='linear').fit(datax,data_y))
    return svms



def predict(svms,x):
    result=np.zeros(N)
    for svm in svms:
        predict=svm.predict(x.reshape(1,-1))
        result[int(predict)]+=1
    #print(result)
    return list(result).index(max(result))

def score(svms,testx,testy):
    n=len(testy)
    cpt=0
    for i in range(n):
        y_predict=predict(svms,testx[i,:])
        if(y_predict==y[i]):
            cpt+=1
    return cpt/n
