from arftools import *
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib import cm

def one_versus_one(datax,datay,testx,testy):
	svms=[]

	y_=np.unique(datay)
	y1_=[]
	y2_=[]
	for y1 in y_:
		for y2 in y_:
			if(y1!=y2)&(y1<y2):
				#division des donnees
				trainx,trainy=datax[(datay==y1)|(datay==y2)],datay[(datay==y1)|(datay==y2)]
				testx,testy=testx[(datay==y1)|(datay==y2)],testy[(datay==y1)|(datay==y2)]
	print(y1,y2)
