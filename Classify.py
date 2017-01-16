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

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('Aut_classify.mat')
x = dataset['data']
y = dataset['y']

# print x.shape, y.shape
kf_total = cross_validation.KFold(len(x), n_folds=50, shuffle=False, random_state=782828)
# kf_total = StratifiedKFold(n_splits =10)
# kf_total.get_n_splits(x, y)

# for train, test in kf_total:
#  	print x[train], '\n', np.asarray(y[train]), '\n\n'

sklearn_pca = sklearnPCA(n_components=167)
sklearn_kpca = sklearnKPCA(n_components=10	,kernel="rbf")
clf = SVC(kernel ='rbf',C=0.07)
clf2 = SVC(kernel ='rbf',C=0.07)
clf3 = SVC(kernel ='rbf',C=0.07)

# for train, test in kf_total.split(x,y):
# 	sklearn_transf = sklearn_pca.fit_transform(x[train])
# 	# print sklearn_pca.explained_variance_ratio_
# 	# print sklearn_transf.shape
# 	red_data_test = sklearn_pca.transform(x[test])
# 	# print red_data_test.shape
# 	clf.fit(sklearn_pca.fit_transform(x[train]),np.asarray(y[train],dtype = np.float32))
# 	print " accuracy on reduced dataset using PCA \n"
# 	print clf.score(sklearn_pca.transform(x[test]),np.asarray(y[test],dtype = np.float32))
# 	clf2.fit(x[train],np.asarray(y[train],dtype = np.float32))
# 	print " Baseline Accuracy\n"
# 	print clf2.score(x[test],np.asarray(y[test],dtype = np.float32))
#     clf3.fit(sklearn_kpca.fit_transform(x[train]),np.asarray(y[train],dtype = np.float32))
#     print " accuracy on reduced dataset using KPCA \n"
#     print clf3.score(sklearn_kpca.transform(x[test]),np.asarray(y[test],dtype = np.float32))
# #     c_range = np.logspace(0, 4, 10)
     
 
# lrgs = grid_search.GridSearchCV(estimator=clf, param_grid=dict(C=c_range), n_jobs=1)
print " accuracy on reduced dataset using PCA \n"
print [clf.fit(sklearn_pca.fit_transform(x[train_indices]),np.asarray(y[train_indices],dtype = np.float32)) \
      .score(sklearn_pca.transform(x[test_indices]),np.asarray(y[test_indices],dtype = np.float32)) \
	  for train_indices, test_indices in kf_total]

print " Baseline \n"
print [clf2.fit(x[train_indices],np.asarray(y[train_indices],dtype = np.float32)) \
      .score(x[test_indices],np.asarray(y[test_indices],dtype = np.float32)) \
	  for train_indices, test_indices in kf_total]

print " accuracy on reduced dataset using kPCA \n"    
print [clf.fit(sklearn_kpca.fit_transform(x[train_indices]),np.asarray(y[train_indices],dtype = np.float32)) \
      .score(sklearn_kpca.transform(x[test_indices]),np.asarray(y[test_indices],dtype = np.float32)) \
	  for train_indices, test_indices in kf_total]
