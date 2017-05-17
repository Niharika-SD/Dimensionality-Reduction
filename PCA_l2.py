import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob,os
from pcp_outliers import pcp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn import metrics,cross_validation
from sklearn.decomposition import PCA,MiniBatchSparsePCA,RandomizedPCA
from sklearn.svm import SVC,NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from pylab import * 
import scipy.io as sio
import pandas as pd
from SparsePCA_Pipeline import Split_class,create_dataset,evaluate_results

df_aut,df_cont = Split_class()
task  ='ADOS.Total'
x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')	
	
x= x_aut
y= np.ravel(y_aut)

pca =PCA()
pca1 = PCA(n_components = 39)

x = pca.inverse_transform(pca.fit_transform(x))- pca1.inverse_transform(pca1.fit_transform(x))
svr_poly = NuSVR(kernel ='rbf',cache_size=1000)

kf_total = KFold(n_splits=10,shuffle=True, random_state=78)

ridge_range = np.linspace(0.01,0.09,5)
c_range = np.logspace(0,5,5)
gamma_range = np.logspace(-4, 1, 6)
my_scorer = make_scorer(explained_variance_score)
p_grid = dict(C =c_range,nu=ridge_range, gamma = gamma_range)


lrgs = GridSearchCV(estimator=svr_poly, param_grid=p_grid, scoring =my_scorer, n_jobs=1)
lrgs.fit(x,y)
print [explained_variance_score(y[test],lrgs.fit(x[train],y[train]).predict(x[test]),multioutput='variance_weighted') for train, test in kf_total.split(x,y)]
print [mean_absolute_error(lrgs.fit(x[train],y[train]).predict(x[test]),y[test]) for train, test in kf_total.split(x,y)]
print lrgs.best_score_
print lrgs.best_estimator_

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/'+ task
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)
evaluate_results(kf_total,x,y,lrgs.best_estimator_,-1)
