from arftools import *
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib import cm



def plot_svc_decision_frontiere(model,ax=None,data=None):
	step=20
	if(ax is None):
		ax=plt.gca()
	xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
	x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    #grid=np.c_[x.ravel(),y.ravel()]
	#x=np.linspace(xlim[0],xlim[1],30)
	#y=np.linspace(ylim[0],ylim[1],30)
	Y,X=np.meshgrid(y,x)
	xy=np.vstack([X.ravel(),Y.ravel()]).T
	P=model.decision_function(xy).reshape(X.shape)
	ax.contour(X,Y,P,colors="k",levels=[-1,0,1],alpha=0.5)
	#ax.set_xlim(xlim)
	#ax.set_ylim(ylim)


def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def plot_frontiere_proba(data,f,step=20):
	grid,x,y=make_grid(data=data,step=step)
	plt.contourf(x,y,f(grid).reshape(x.shape),255)


def plot_erreur(gamma_range,kernel,trainx,trainy,testx,testy):
	score=[]
	for i in gamma_range:
		clf = SVC(kernel=kernel,gamma=i).fit(trainx,trainy)
		score.append(clf.score(testx,testy))
	print(max(score),gamma_range[score.index(max(score))])
	plt.plot(gamma_range,score)


def grid_search(gamma_range,coef_range,trainx,trainy,testx,testy):
	params_grid=dict(gamma=gamma_range,coef0=coef_range)
	

	cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=420)

	clf=GridSearchCV(SVC(kernel="poly",degree=1),param_grid=params_grid,cv=cv)
	clf.fit(trainx,trainy)
	return(clf.best_params_,clf.best_score_)
'''
def plot_3D(elev=30,azim=30,X=X,y=y):
	ax=plt.subl=plot(projection='3d')
	ax.scatter3D(X[:,0],X[:,1],r,c=y,camp="rainbow")
	ax.view_init(elev=elev,azim=azim)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	plt.show()
'''


def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data=[[float(x) for x in l.split()] for l in f if len(l.split())>2]
        tmp=np.array(data)
        return tmp[:,1:], tmp[:,0].astype(int)