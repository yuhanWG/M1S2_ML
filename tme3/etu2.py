from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    return np.mean((datax.dot(w.T)-datay)**2)

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    n,d=datax.shape
    return (2*datax.T).dot((datax.dot((w.T)-datay)))
    #retrouner les adjustement des poids:donc taille d*1

def hinge(datax,datay,w):
    """retourne la moyenne de l'erreur hinge"""
    #hinge: si bien classifie, -y<w.x><0 sinon >0. On ne compte celui sont mal-classifies
    fx = np.dot(datax, w.T)
    return np.mean(np.where(-datay*fx<0,0,-datay*fx))

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    #chaque exemplaire donne un gradient
    fx=-datay*np.dot(datax,w.T)
    fx2=-datay.reshape(-1,1)*datax
    res=[]
    for i in range(len(fx)):
        if fx[i]<0:
            res.append(np.zeros(len(w)))
        else:
            res.append(fx2[i])
    return np.mean(np.array(res), axis=0)


    if len(datax.shape)==1:
        datax = datax.reshape(1,-1)
    n,d=datax.shape
    yx=[-datay[i].reshape(1,-1)*datax[i,:] if -datay[i]*np.dot(datax[i,:],w.T)>0 else np.zeros(d) for i in range(n)]

    return np.mean(yx,axis=0)

def hinge_stochastique(datax,datay,w):
    n,d=datax.shape
    r=np.random.randint(0,datax.shape[0], size=None, dtype='l')
    return max(0,-datay[r]*np.dot(datax[r,:],w.T))
    
def hinge_g_stochastique(datax,datay,w):
    #prend au hasard un exemple
    n,d=datax.shape
    r=np.random.randint(0,datax.shape[0], size=None, dtype='l')
    if(-datay[r]*np.dot(datax[r,:],w.T)>0):
        return -datay[r].reshape(1,-1)*datax[r,:]
    else:
        return np.zeros(d)

#si on voudrais ajouter un biais
def datax_with_biais(datax):
    if(len(datax.shape)==1):
        datax = datax.reshape(1,-1)
    n,d=datax.shape
    return np.hstack((datax,np.ones(n).reshape(-1,1)))


class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
    #def __init__(self,loss=hinge,loss_g=hinge_g,max_iter,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        #self.w = np.zeros((1,D))
        for i in range(self.max_iter):
            self.w = self.w - self.eps*self.loss_g(datax, datay, self.w)
        #pass

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.sign(np.dot(datax,self.w.reshape(-1,1))).reshape(1,-1)[0]

    def score(self,datax,datay):
        pre = self.predict(datax)
        res = np.where(pre==datay,1,0)
        return np.sum(res)/len(res)
        #return self.loss(datax,datay, self.w)
    
def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data=[[float(x) for x in l.split()] for l in f if len(l.split())>2]
        tmp=np.array(data)
        return tmp[:,1:], tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()


def project_polynomiale(datax):
    if(datax.shape[1]==2):
        x1x2=datax[:,0]*datax[:,1]
        x12=datax[:,0]**2
        x22=datax[:,1]**2

        x=np.hstack((datax,x1x2))
        x=np.hstack((x,x12))
        x=np.hstack((x,x22))

        return x
