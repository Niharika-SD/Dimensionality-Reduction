import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn import linear_model,grid_search,cross_validation
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('Aut_classify.mat')
x = dataset['data']
y = dataset['y']
y = np.ravel(y)

# print x.shape, y.shape
# kf_total = cross_validation.KFold(len(x), n_folds=50, shuffle=False, random_state=782828)
kf_total = StratifiedKFold(n_splits =10,shuffle=True)
kf_total.get_n_splits(x, y)

for train, test in kf_total.split(x,y):
 	print test 
 	print x[train].shape, '\n', y[train].shape

sklearn_pca = sklearnPCA()
sklearn_kpca = sklearnKPCA(kernel="poly")

clf = SVC(kernel ='rbf')
clf2 = SVC(kernel ='linear')
clf3 = SVC(kernel ='rbf')

pca_svm = Pipeline([('pca',sklearn_pca), ('svc', clf)])
kpca_svm = Pipeline([('kpca',sklearn_kpca), ('svc', clf3)])

c_range = np.logspace(-2, 2, 4)
n_comp = np.linspace(5, 30, num=7,dtype = 'int32')
     
 
# print " accuracy on reduced dataset using PCA \n"
# print [clf.fit(sklearn_pca.fit_transform(x[train_indices]),np.asarray(y[train_indices],dtype = np.float32)) \
#       .score(sklearn_pca.transform(x[test_indices]),np.asarray(y[test_indices],dtype = np.float32)) \
# 	  for train_indices, test_indices in kf_total.split(x,y)]

print " Baseline \n"
print np.mean([clf2.fit(x[train_indices],np.asarray(y[train_indices],dtype = np.float32)) \
      .score(x[test_indices],np.asarray(y[test_indices],dtype = np.float32)) \
	  for train_indices, test_indices in kf_total.split(x,y)])

print " accuracy on reduced dataset using PCA \n"
lrgs = grid_search.GridSearchCV(estimator=pca_svm, param_grid=dict(pca__n_components = n_comp), n_jobs=1)
print [lrgs.fit(x[train],y[train]).score(x[test],y[test]) for train, test in kf_total.split(x,y)]
print lrgs.best_score_
print lrgs.best_estimator_

print " accuracy on reduced dataset using kPCA \n"
lrgs = grid_search.GridSearchCV(estimator=kpca_svm, param_grid=dict(kpca__n_components = n_comp,kpca__gamma = c_range), n_jobs=1)
print [lrgs.fit(x[train],y[train]).score(x[test],y[test]) for train, test in kf_total.split(x,y)]
print lrgs.best_score_
print lrgs.best_estimator_