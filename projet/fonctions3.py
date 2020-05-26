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
	on fixe h est impair
	'''
	n=img.shape[1]
	d=int((h-1)/2)

	if(i-d>=0)&(i+d<n)&(j-d>=0)&(h+d<n):
		return img[i-d:i+d+1,j-d:j+d+1,:]
	else:
		pass
		#raise exception value error

def patch_to_vector(patch):
	vecteur=np.reshape(patch,(-1,1))
	return vecteur


def vector_to_patch(vector,h,d=3):
	return vector.reshape(h,h,d)


def noise(img,prc):
	'''
	supprimer au hasard un pourcentage de pixel dans l'image
	pour les pixels supprimes, on les exprime avec valeur -100
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
		'''
		for j in range(d):
			data_noise[x,y,j]=-100
		'''
		data_noise[x,y]=[-100,-100,-100]
	
	#return note,data_noise
	return data_noise



def delete_rect(img,i,j,height,width):
	'''
	supprimer tout un rectangle de l’image
	'''
	n,n,d=img.shape
	#note=np.ones((n,n))
	data_rect=img.copy()
	
	data_rect[i:i+height,j:j+width]=np.ones((height,width,3))*(-100)
	#return note,data_rect
	return data_rect
	#pass

def dict_maquant(img,h):
	patch_manquant=[]
	dictionnaire=[]
	pos=[]

	d=int((h-1)/2)
	n=img.shape[0]
	for i in range(d,n-d,h):
		for j in range(d,n-d,h):
			patch=get_patch(i,j,h,img)
			vector=patch_to_vector(patch)
			if(np.any(vector==-100)):
				patch_manquant.append(patch)
				pos.append((i,j))
			else:
				dictionnaire.append(patch)

	return patch_manquant,dictionnaire,pos


def debruiter(patch,d,alpha=0.00001,max_iter=1000):
	lasso=linear_model.Lasso(alpha=alpha,max_iter=max_iter)
	datay=patch_to_vector(patch)
	index=np.where(datay!=-100)
	datay=datay[index]
	if(len(datay)==0):
		return None
	else:
		#print(patch_to_vector(d[0]))
		datax=np.array([patch_to_vector(patch)[index] for patch in d]).T
		lasso.fit(datax, datay)
	return lasso


def inpainting(img,h):
	patch_manquant,dt,pos=dict_maquant(img,h)


	print(len(patch_manquant))
	print(len(dt))
	print(len(pos))

	predict=[]
	repair=img.copy()
	d=int((h-1)/2)
	datax=np.array([patch_to_vector(patch) for patch in dt]).T[0]
	#print("datax",datax.shape)

	for pm in patch_manquant:
		lasso=debruiter(pm,dt)
		if(lasso!=None):
			vpredict=lasso.predict(datax)
			predict.append(vector_to_patch(vpredict,h))
		else:
			predict.append(pm)

	#reconstruction
	for i in range(len(pos)):
		x=pos[i][0]
		y=pos[i][1]
		repair[x-d:x+d+1,y-d:y+d+1,:]=predict[i]

	return repair
