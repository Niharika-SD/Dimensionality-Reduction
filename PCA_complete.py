import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
import scipy.io as sio
import os

os.chdir('/home/niharikashimona/MLSP-Project/patient_data/')

data = sio.loadmat('data_1.mat')
data_train = data['data_train']
data_test = data['data_test']

sklearn_pca = sklearnPCA(n_components=10)
sklearn_transf = sklearn_pca.fit_transform(data_train)
print(sklearn_pca.explained_variance_ratio_)

red_data_test = sklearn_pca.transform(data_test)
print(red_data_test.shape)

sklearn_kpca = sklearnKPCA(n_components=10,kernel="rbf")
sklearn_transf_kpca = sklearn_kpca.fit_transform(data_train)
red_data_test = sklearn_kpca.transform(data_test)
print(red_data_test.shape)
