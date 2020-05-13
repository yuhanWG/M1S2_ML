import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))


## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
print(len(poidata[typepoi]))
print(geo_mat.shape)
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]
print(geo_mat[0,:])
print(np.floor(5/2))
## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################
def histogramme(xx,yy,geo_mat,steps):
	xstep=(xmax-xmin)/steps
	ystep=(ymax-ymin)/steps

	hist=np.zeros((steps,steps))
	for i in range(geo_mat.shape[0]):
			#obtenir une coordonne d'un point de POI
		y,x=geo_mat[i,:]
		'''print(y,x)
								print(y-ymin,x-xmin)'''
		argx=np.int(np.floor((x-xmin)/xstep))
		argy=np.int(np.floor((y-ymin)/ystep))

		'''print(ystep)
								print(xstep)
								
								print(y-ymin/ystep,x-xmin/xstep)
								'''
		hist[argx,argy]+=1
	#normalisation
	hist=[h/geo_mat.shape[0] for h in hist]
	print(type(hist))
	return np.matrix(hist)

def predict(px,py,hist,steps):
	#retourner la densite
	xstep=(xmax-xmin)/steps
	ystep=(ymax-ymin)/steps
	print(xstep,ystep)
	print("PX",px,py)
	#print((px-xmin)/xstep)
	#print((py-ymin)/ystep)
	argx=np.int(np.floor((px-xmin)/xstep))
	argy=np.int(np.floor((py-ymin)/ystep))
	return hist[argy,argx]

def phi_uniforme(p,pi,h):
	if((p[0]-pi[0])/h>1/2) or ((p[1]-pi[1])/h>1/2):
		return 0
	else:
		return 1

def noyaux(xx,yy,geo_mat,h):
	d=geo_mat.shape[1]
	V=h**d
	for x in geo_mat:
		kn=





# discretisation pour l'affichage des modeles d'estimation de densite
steps = 30
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
print(xx.shape,yy.shape)

grid = np.c_[xx.ravel(),yy.ravel()]
#grid represente les coordinations x,y
print(grid.shape)

# A remplacer par res = monModele.predict(grid).reshape(steps,steps)
#res = np.random.random((steps,steps))
res=histogramme(xx,yy,geo_mat,steps)
# res est une matrice de random (0,1) a shape (steps,steps)
#print(res)
print("hello",predict(2.31,48.875,res,steps))
plt.figure()
show_map()
plt.imshow(res.T,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
#plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.3)
input("press enter")
