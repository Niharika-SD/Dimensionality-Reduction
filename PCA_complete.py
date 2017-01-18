import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model,cross_validation
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('bytSrsPSocTotSrsRaw.mat')
x= dataset['data']
y = dataset['y']
y = np.ravel(y)
print y.shape

kf_total = cross_validation.KFold(len(x), n_folds=10, shuffle=False, random_state=782828)
sklearn_pca = sklearnPCA(n_components=5)
sklearn_kpca = sklearnKPCA(n_components=10	,kernel="rbf")

lr = linear_model.LinearRegression()

print " accuracy on reduced dataset using PCA \n"
print [lr.fit(sklearn_pca.fit_transform(x[train_indices]),y[train_indices]) \
      .score(sklearn_pca.transform(x[test_indices]),y[test_indices]) \
	  for train_indices, test_indices in kf_total]

# print " Baseline \n"
# print [lr.fit(x[train_indices],y[train_indices]) \
#       .score(x[test_indices],y[test_indices]) \
# 	  for train_indices, test_indices in kf_total]

print " accuracy on reduced dataset using kPCA \n"    
print [lr.fit(sklearn_kpca.fit_transform(x[train_indices]),y[train_indices]) \
      .score(sklearn_kpca.transform(x[test_indices]),y[test_indices]) \
	  for train_indices, test_indices in kf_total]

