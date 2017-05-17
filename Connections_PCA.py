from time import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import SparsePCA
import sys,os
from pcp_outliers import pcp
iter =0
os.chdir('/home/niharika-shimona/Documents/Projects/Autism_Network/code/Datasets_Matched')
dataset = sio.loadmat('Autism_cl.mat')

if iter== 0:
	x = dataset['data_Aut']
	id = dataset['id_Aut']
	y = dataset['y_aut']
	
	# sys.stdout=open('PCA_connect_Aut.txt',"w")
elif iter==1:
	x = dataset['data_Controls']
	id = dataset['id_con']
	y = dataset['y_con']
	
	# sys.stdout=open('PCA_connect_Controls.txt',"w")
else :
	
	y = np.concatenate((dataset['y_aut'],dataset['y_con']),axis =0)
	id = np.concatenate((dataset['id_Aut'],dataset['id_con']),axis =0)
	x = np.concatenate((dataset['data_Aut'],dataset['data_Controls']),axis =0)
	
	# sys.stdout=open('PCA_connect_Complete.txt',"w")


# sklearn_pca= sklearnPCA(n_components =10)
# sklearn_pca  = SparsePCA(n_components=30, alpha=5, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=1, U_init=None, V_init=None, verbose=False, random_state=None)
# sklearn_trasf_pca = sklearn_pca.fit_transform(x)

# # X = sklearn_pca.inverse_transform(sklearn_trasf_pca)- np.ones((x.shape))
# # print X
# y = np.dot(np.linalg.pinv(sklearn_pca.components_),sklearn_trasf_pca.T)
# k =0
L, E, (u, s, v)  = pcp(x.T, maxiter=500, verbose=True, svd_method="exact")
L = L.T
E = E.T
# for i in range X.shape[0]:
# 	for j in range X.shape[1]:
		
# 		Corr[i] = np.identity(116)
# 		l =116 -j-1
# 		for 
# 			Corr[i][k][l]



sio.savemat('lowrank.mat',{'L': L})
sio.savemat('outliers.mat',{'E': E})