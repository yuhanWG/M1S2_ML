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


def bruiter(img,prc):
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
	note=np.ones((n,n))
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
	dictionnaire=np.array([patch_to_vector(patch) for patch in dt]).T[0]
	#print("datax",datax.shape)

	for pm in patch_manquant:
		lasso=debruiter(pm,dt)
		if(lasso!=None):
			vpredict=lasso.predict(dictionnaire)
			predict.append(vector_to_patch(vpredict,h))
		else:
			predict.append(pm)

	#reconstruction
	for i in range(len(pos)):
		x=pos[i][0]
		y=pos[i][1]
		repair[x-d:x+d+1,y-d:y+d+1,:]=predict[i]

	return repair


def delete_rect2(img,i,j,height,width):
	'''
	supprimer tout un rectangle de l’image, retourner les images apres la supression 
    et la matrice qui enregistre confidence C pour les pixels.
    Pour l'initialisation
    C(p)=1 ou 0
	'''
	n,n,d=img.shape
	note=np.ones((n,n))
	data_rect=img.copy()
	
	data_rect[i:i+height,j:j+width]=np.ones((height,width,3))*(-100)
	note[i:i+height,j:j+width]=np.ones((height,width))*0
	#return note,data_rect
	return data_rect,note

def patch_C(C,h,i,j,height,width):
    n=C.shape[0]
    d=int((h-1)/2)
    patchC=C.copy()
    
    for i in range(d,n-d):
        for j in range(d,n-d):
            #ndarray to 1d
            l=C[i-d:i+d+1,j-d:j+d+1].ravel()
            index=np.where(l==1)
            #les points appartiennet a la source region
            patchC[i,j]=np.sum(l[index])/(h*h)
    return patchC

def filling(img,h,C,i,j,height,width):
    '''
    img: image a repair, taille (n,n,d)
    h: longeur du patch
    C: confidence initial, taille (n,n)
    '''
    repair=img.copy()
    n=img.shape[0]
    d=int((h-1)/2)
    
    #get dictionnaire
    patch_manquant,dt,pos=dict_maquant(img,h)
    dictionnaire=np.array([patch_to_vector(patch) for patch in dt]).T[0]
    #initialisation 
    
    patchC=patch_C(C,h,i,j,height,width)
    print(np.unique(patchC))
    cpt=0
    test=[]
    #While il reste encore les pixels a remplir
    while(C.all()!=1):
    #while(len(np.where(repair==-100))!=0):
        #find the patch with the maximun priority sauf 1
        maxPriority=max(np.unique(patchC[patchC!=1]))
        maxPriority=np.unique(patchC)[-2]
        #print(maxPriority)
        
        xx,yy=np.where(patchC==maxPriority)
        afill=np.random.randint(0,len(xx))
        x,y=xx[afill],yy[afill]
        #print(x,y)
        #remplir ce patch
        patch=repair[x-d:x+d+1,y-d:y+d+1]
        lasso=debruiter(patch,dt)
        if(lasso!=None):
            vpredict=lasso.predict(dictionnaire)
            predict=vector_to_patch(vpredict,h)
            #print(predict.shape,repair[x-d:x+d+1,y-d:y+d+1].shape)
            repair[x-d:x+d+1,y-d:y+d+1]=predict
            nb1=len(np.where(C==0)[0])
        #update C
            C[x-d:x+d+1,y-d:y+d+1]=np.ones((h,h))
            nb2=len(np.where(C==0)[0])
            #print(nb1-nb2)
            test.append(nb1-nb2)
            #update patch_C
            patchC=patch_C(C,h,i,j,height,width)
            
        else:
            #print("error")
            cpt+=1
    
    print("cpt",cpt)
    return repair,C,patchC
