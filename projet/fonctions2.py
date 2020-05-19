#ce fichier enregistre les fonctions dont on a besoin
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import math
import random


def read_im(fn):
	'''
	retourner image apres normalisation
	'''
	data=plt.imread(fn).copy()
	return data/255


def get_patch(i,j,h,img):
	'''
	retourner le patch centre au (i,j) et de longeur h d'une image im
	'''
	n,n,d=img.shape
	d=int((h-1)/2)

	if(i-d>=0)&(i+d<n)&(j-d>=0)&(h+d<n):
		return img[i-d:i+d+1,j-d:j+d+1,:]
	else:
		pass
		#raise exception value error

def patch_to_vector(patch):
	vecteur=np.reshape(patch,(1,-1))
	return vecteur


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


def patch_pixel_maquant(img,matrice_note,h):
	'''
	renvoyer les patchs qui contient les pixels maquants
	'''
	pixel_maquant=[]
	n,n,d=img.shape
	d=int((h-1)/2)
	for i in range(d,n-d):
		for j in range(d,n-d):
			traite=get_patch(i,j,h,img)
			note=matrice_note[i-d:i+d+1,j-d:j+d+1]
			#if not(np.all(matrice_note[i-d:i+d+1,j-d:j+d+1])==1):
			if not(len(np.unique(note))==1)&(np.unique(note)[0]==1):
				pixel_maquant.append(patch_to_vector(traite))
			
	return np.array(pixel_maquant)

def dictionnaire(img,matrice_note,h):
	'''
	renvoyer les patchs qui ne contient aucun pixel maquant
	'''
	#pixel_complet=[]
	n,n,d=img.shape
	pixel_complet=[]


	d1=int((h-1)/2)
	for i in range(d1,n-d1):
		for j in range(d1,n-d1):
			a=get_patch(i,j,h,img)
			#traite=patch_to_vector(a)
			b=patch_to_vector(a)
			note=matrice_note[i-d1:i+d1+1,j-d1:j+d1+1]
			if (len(np.unique(note))==1)&(np.unique(note)[0]==1):
				#np.vstack((pixel_complet,b))
				pixel_complet.append(b)
	
	return np.array(pixel_complet).reshape(-1,h*h*d)
	#return np.delete(pixel_complet,0,0)
	#pass

def dictionnaire_patch(dictionnaire,patch,alpha):
	'''
	patch: k dimensionnelle vecteur avec centre pm

	'''
	lasso=linear_model.Lasso(alpha=alpha)
	d=dictionnaire.shape[1]
	for i in range(dictionnaire.shape[0]):
		lasso.fit(dictionnaire.T,dictionnaire[i,:])
	return lasso.coef_
