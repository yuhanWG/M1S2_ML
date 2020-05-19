#ce fichier enregistre les fonctions dont on a besoin
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import math
import random

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



######Partie inpainting#####
def read_im(fn):
	'''
	retourner image apres normalisation
	'''

	data=plt.imread(fn).copy()
	return data/255
	#return colors.rgb_to_hsv(data/255)


def get_patch(i,j,h,im):
	'''
	retourner le patch centre au (i,j) et de longeur h d'une image im
	'''
	pass

def noise(img,prc):
	'''
	supprimer au hasard un pourcentage de pixel dans l'image
	'''
	#pass
	data_noise=img.copy()
	n,n,d=img.shape
	nb_noise=int(math.ceil((n**2)*prc))
	note=np.ones((n,n))

	for i in range(nb_noise):
		x=random.randint(0,n-1)
		y=random.randint(0,n-1)
		while(note[x,y]==-100):
			x=random.randint(0,n-1)
			y=random.randint(0,n-1)
		note[x,y]=-100
		for j in range(d):
			data_noise[x,y,j]=0
	
	return note,data_noise


def delete_rect(img,i,j,height,width):
	'''
	supprimer tout un rectangle de l’image
	'''
	n,n,d=img.shape
	note=np.ones((n,n))
	data_rect=img.copy()
	for a in range(height):
		for b in range(width):
			for c in range(d):
				data_rect[i+a,j+b,c]=-100
			note[i+a,j+b]=-100
	return note,data_rect
	#pass


def patch_pixel_maquant(img):
	'''
	renvoyer les patchs qui contient les pixels maquants
	'''
	pass

def dictionnaire(img):
	'''
	renvoyer les patchs qui ne contient aucun pixel maquant
	'''
	pass

