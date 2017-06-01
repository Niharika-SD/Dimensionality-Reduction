import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob,os
from pcp_outliers import pcp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA,MiniBatchSparsePCA,RandomizedPCA
from sklearn import metrics,cross_validation
from sklearn.svm import SVC,NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from pylab import * 
import scipy.io as sio
import pandas as pd
from Data_Extraction import Split_class,create_dataset,evaluate_results

df_aut,df_cont = Split_class()
task  = sys.argv[1]
ds = sys.argv[2]
cas = sys.argv[3]
x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data/')	

if ds == '0':	
	x = x_cont
	y= np.ravel(y_cont)
	fold = 'cont'
	n_comp = 38
elif ds == '1':
	x = x_aut
	y= np.ravel(y_aut)
	fold = 'aut'
	n_comp = 38
else:
	x =np.concatenate((x_cont,x_aut),axis =0)
	y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))
	fold = 'aut_cont'
	n_comp = 76

pca =PCA()
pca1 = PCA(n_components = n_comp)
print 'hi'

if cas =='0':
	x_comp = x
	cast = 'whole' 
elif cas == '1':
	x_comp = pca.inverse_transform(pca.fit_transform(x))-pca1.inverse_transform(pca1.fit_transform(x))
	cast = 'Rank_Subtracted'
elif cas =='2':
	x_comp = pca.fit_transform(x)[:,1:n_comp]
	cast = 'Reduced_Rank'
elif cas == '3':
	L,E,(u,s,v) = pcp(x,'gross_errors', maxiter=1000, verbose=True, svd_method="exact",)
	x_comp =E
	cast = 'Robust_PCA'

svr = NuSVR(C=17.782794100389228, cache_size=1000, coef0=0.0, degree=3,
   gamma=0.0001, kernel='linear', max_iter=-1, nu=0.069999999999999993,
   shrinking=True, tol=0.001, verbose=False)
ridge_range = np.linspace(0.01,0.09,5)
c_range = np.logspace(0,5,5)

kf_total = KFold(n_splits=10,shuffle=False, random_state=6)
p_grid = dict(C =c_range, nu=ridge_range)
my_scorer = make_scorer(explained_variance_score)

y_train =[]
y_test = []
y_train_AF =[]
y_test_AF =[]
r2_test = []
mse_test =[]

i = 0

for train,test in kf_total.split(x_comp,y):
	
	# lrgs = GridSearchCV(estimator=svr, param_grid=p_grid, scoring =my_scorer, n_jobs=-1)
	# lrgs.fit(x_comp[train],y[train])
	model = svr.fit(x_comp[train],y[train])
	y_pred_train = model.predict(x_comp[train])
	y_pred_test = model.predict(x_comp[test])
	r2_test.append(r2_score(y[train],y_pred_train))
	mse_test.append(mean_squared_error(y[train],y_pred_train))

	y_train_AF = np.concatenate((y_train_AF,y_pred_train),axis =0)
	y_train = np.concatenate((y_train,y[train]),axis =0)
	y_test_AF = np.concatenate((y_test_AF,y_pred_test),axis =0)
	y_test = np.concatenate((y_test,y[test]),axis =0)
		
	i = i +1

newpath = r'/home/niharika-shimona//Documents/Projects/Autism_Network/Results/Sanity_Check/svr/'+ fold + '/'+ cast +'/' +task + '/'
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)

fig,ax =plt.subplots()
ax.scatter(y_train,y_train_AF,color ='red')
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
name = 'fig_train.png'
fig.savefig(name)   # save the figure to fil
plt.close(fig)

fig,ax =plt.subplots()
ax.scatter(y_test,y_test_AF,color ='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.scatter(y_test,y_test_AF,color ='green')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
name = 'fig_test.png'
fig.savefig(name)   # save the figure to fil
plt.close(fig)


pickle.dump(model, open('model.p', 'wb'))
sio.savemat('Metrics_full.mat',{'r2_test':r2_test,'mse_test':mse_test})