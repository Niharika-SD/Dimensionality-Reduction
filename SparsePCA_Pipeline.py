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
from sklearn import metrics
from sklearn.decomposition import PCA,MiniBatchSparsePCA,RandomizedPCA
from sklearn.svm import SVC,NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from pylab import * 
import scipy.io as sio
import pandas as pd

def Split_class():
	
	"Splits the dataset into Autism and Controls"

	Location = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Data/matched_data.xlsx'
	df = pd.read_excel(Location,0)
	mask_cont = df['Primary_Dx'] == 'None' 
	mask_aut = df['Primary_Dx'] != 'None' 

 	df_cont = df[mask_cont]
 	df_aut = df[mask_aut]
	
	return df_aut,df_cont

def evaluate_results(inner_cv,x,y,final_model,ind):

	"Performs a complete plot based evaluation of the run"    
	i =0
	sPCA_MSE =[]
	sPCA_r2=[]
	sPCA_exp=[]
	sPCA_MSE_test=[]
	sPCA_r2_test =[]
	sPCA_exp_test=[]

	for train, test in inner_cv.split(x,y):

		y_pred_train = np.asarray(final_model.predict(x[train]))
		y_pred_test = np.asarray(final_model.predict(x[test]))
		if ind > -1:
			y_pred_test =y_pred_test[:,ind]
			y_pred_train =y_pred_train[:,ind]
		
		sPCA_MSE.append(mean_squared_error(y[train], y_pred_train))
		sPCA_r2.append(r2_score(y[train], y_pred_train,multioutput='variance_weighted'))
		sPCA_exp.append(explained_variance_score(y[train], y_pred_train,multioutput='variance_weighted'))
		i= i+1
		print 'Split', i ,'\n' 
		print 'MSE : ', mean_squared_error(y[train], y_pred_train)
		print 'Explained Variance Score : ', explained_variance_score(y[train], y_pred_train,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[train], y_pred_train,multioutput='variance_weighted')
		fig, ax = plt.subplots()
		ax.scatter(y[train],y_pred_train)
		ax.plot([y[train].min(), y[train].max()], [y[train].min(), y[train].max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		
		name = 'fig_'+ `i`+ '_train.png'
		fig.savefig(name)   # save the figure to fil
		plt.close(fig)

		sPCA_MSE_test.append(mean_squared_error(y[test], y_pred_test))
		sPCA_r2_test.append(r2_score(y[test], y_pred_test,multioutput='variance_weighted'))
		sPCA_exp_test.append(explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted'))
		print 'MSE : ', mean_squared_error(y[test], y_pred_test)
		print 'Explained Variance Score : ', explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[test], y_pred_test,multioutput='variance_weighted')
		fig, ax = plt.subplots()
		ax.scatter(y[test],y_pred_test)
		ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		name = `i`+ '_test.png'
		fig.savefig(name)   # save the figure to file
		plt.close(fig)

	print(np.mean(sPCA_MSE),np.mean(sPCA_r2),np.mean(sPCA_exp))
	print(np.mean(sPCA_MSE_test),np.mean(sPCA_r2_test),np.mean(sPCA_exp_test))

	return

def create_dataset(df_aut,df_cont,task,folder):
	
	"Creates the dataset according to a regression task"
	
	y_aut = np.zeros((1,1))
	y_cont = np.zeros((1,1))
	x_cont = np.zeros((1,6670))
	x_aut = np.zeros((1,6670))

	df_aut = df_aut[df_aut[task]< 7000]
	df_cont = df_cont[df_cont[task]< 7000]

	for ID_NO,score in zip(df_aut['ID'],df_aut[task]):

		filename = folder + '/Corr_' + `ID_NO` + '.mat'
		data = sio.loadmat(filename) 
		x_aut = np.concatenate((x_aut,data['corr']),axis =0)
		y_aut = np.concatenate((y_aut,score*np.ones((1,1))),axis =0)
		
	if (task!= 'ADOS.Total'):
			
		for ID_NO,score in zip(df_cont['ID'],df_cont[task]):

			filename = folder + '/Corr_' + `ID_NO` + '.mat'
			data = sio.loadmat(filename) 
			x_cont = np.concatenate((x_cont,data['corr']),axis =0)
			y_cont = np.concatenate((y_cont,score*np.ones((1,1))),axis =0)
			

	return x_aut[1:,:],y_aut[1:,:],x_cont[1:,:],y_cont[1:,:]

if __name__ == '__main__':

	df_aut,df_cont = Split_class()
	task  ='SRS.TotalRaw.Score'
	x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')	
	
	x =np.concatenate((x_cont,x_aut),axis =0)
	y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))

	L,E,(u,s,v) = pcp(x,'gross_errors', maxiter=1000, verbose=True, svd_method="exact",)
	E = E
	L = L

	pca = PCA(svd_solver ='arpack')
	svr_poly = NuSVR(kernel ='rbf',cache_size=1000)
	
	spca_svr = Pipeline([('svr', svr_poly)])
	my_scorer = make_scorer(mean_absolute_error)

	ridge_range = np.linspace(0.01,0.09,5)
	c_range = np.logspace(0,5,5)
	gamma_range = np.logspace(-4, 1, 6)
	n_comp = np.asarray(np.linspace(20,40,5),dtype = 'int8')
	p_grid = dict(svr__C =c_range, svr__nu=ridge_range, svr__gamma = gamma_range)

	model  =[] 
	nested_scores =[]

	for i in range(2):
		inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
		outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)

		clf = GridSearchCV(estimator=spca_svr, param_grid=p_grid, scoring =my_scorer,  cv=inner_cv)
		clf.fit(E,y)
		print '\n'
		print clf.best_score_
		print clf.best_estimator_
		model.append(clf.best_estimator_)

		nested_score = cross_val_score(clf, X=E, y=y, cv=outer_cv,scoring =my_scorer)
		print 'mean of nested scores: ', nested_score.mean()
		nested_scores.append(nested_score.mean())
		
	sys.stdout=open('results'+'.txt',"w")

	m = nested_scores.index(max(nested_scores))
	final_model = model[m]
 	print model[m]
	
	newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/RobustPCA/no_split/in/'+ task
	if not os.path.exists(newpath):
			os.makedirs(newpath)
	os.chdir(newpath)
	evaluate_results(inner_cv,E,y,final_model,-1)
	sys.stdout.close()

	SV  = final_model.named_steps['svr'].support_vectors_
	SV_ind = final_model.named_steps['svr'].support_
	SV_coeff = final_model.named_steps['svr'].coef_
	SV_dualcoeff = final_model.named_steps['svr'].dual_coef_
	# sio.savemat('sv_dualcoeff.mat',{'sv_dualcoeff': SV_dualcoeff})
	# sio.savemat('sv_coeff.mat',{'sv_coeff': SV_coeff})
	# sio.savemat('sv_ind.mat',{'sv_ind': SV_ind})
	# sio.savemat('sv.mat',{'sv': SV})
	sio.savemat('lowrank.mat',{'L': L})
	sio.savemat('outliers.mat',{'E': E})
	        
	