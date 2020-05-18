#ce fichier enregistre les fonctions dont on a besoin
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data=[[float(x) for x in l.split()] for l in f if len(l.split())>2]
        tmp=np.array(data)
        return tmp[:,1:], tmp[:,0].astype(int)

def show_usps(data):
	plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
	plt.colorbar()


def datax_with_biais(datax):
    if(len(datax.shape)==1):
        datax = datax.reshape(1,-1)
    n,d=datax.shape
    return np.hstack((datax,np.ones(n).reshape(-1,1)))


def score(y,y_predict):
	return np.sum(np.where(y==y_predict,1,0))/len(y)


