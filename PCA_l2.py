import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob,os
from pcp_outliers import pcp
from sklearn.ensemble import RandomForestRegressor
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
from Data_Extraction import Split_class,create_dataset,evaluate_results

def visualise_results(inner_cv,x,y,final_model,ind):

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

		y = np.concatenate((y[test],y[train]),axis =0)
		print 'Split', i ,'\n' 
		print 'MSE : ', mean_squared_error(y[train], y_pred_train)
		print 'Explained Variance Score : ', explained_variance_score(y[train], y_pred_train,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[train], y_pred_train,multioutput='variance_weighted')
		sPCA_MSE_test.append(mean_squared_error(y[test], y_pred_test))
		sPCA_r2_test.append(r2_score(y[test], y_pred_test,multioutput='variance_weighted'))
		sPCA_exp_test.append(explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted'))
		print 'MSE : ', mean_squared_error(y[test], y_pred_test)
		print 'Explained Variance Score : ', explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[test], y_pred_test,multioutput='variance_weighted')
		
		fig, ax = plt.subplots()
		lo = ax.scatter(y[train],y_pred_train,color ='red')
		li = ax.scatter(y[test],y_pred_test,color ='green')	
		ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
		ax.legend((lo,li),('train','test'))
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		
		name = 'fig_'+ `i`+ '_train.png'
		fig.savefig(name)   # save the figure to fil
		plt.close(fig)

	print(np.mean(sPCA_MSE),np.mean(sPCA_r2),np.mean(sPCA_exp))
	print(np.mean(sPCA_MSE_test),np.mean(sPCA_r2_test),np.mean(sPCA_exp_test))

	return sPCA_MSE,sPCA_MSE_test,sPCA_exp,sPCA_exp_test,sPCA_r2,sPCA_r2_test

df_aut,df_cont = Split_class()
task  ='SRS.TotalRaw.Score'
x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')	
	
# x =np.concatenate((x_cont,x_aut),axis =0)
# y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))
x = x_aut
y= np.ravel(y_aut)

pca =PCA()
pca1 = PCA(n_components = 38)

x_comp = x 
x_new = pca.inverse_transform(pca.fit_transform(x))-pca1.inverse_transform(pca1.fit_transform(x)) 

rf_reg = RandomForestRegressor(n_estimators=1000,oob_score= True)
rf_reg2 = RandomForestRegressor(n_estimators=1000,oob_score= True)
rf_reg3 = RandomForestRegressor(n_estimators=1000,oob_score= True)

kf_total = KFold(n_splits=10,shuffle=True, random_state=6)
depth_range = np.linspace(10,60,6)
sample_split = np.linspace(0.1,0.5,5)

my_scorer = make_scorer(explained_variance_score)
p_grid = dict(max_depth =depth_range,min_samples_split= sample_split)

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/grid_search/aut/'+ task + '/full/vis/'
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)
sys.stdout=open('results'+'.txt',"w")

print 'Original :'

lrgs = GridSearchCV(estimator=rf_reg, param_grid=p_grid, scoring =my_scorer, n_jobs=-1)
lrgs.fit(x,y)
print [explained_variance_score(y[test],lrgs.fit(x_comp[train],y[train]).predict(x_comp[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_comp,y)]
print [mean_squared_error(lrgs.fit(x_comp[train],y[train]).predict(x_comp[test]),y[test]) for train, test in kf_total.split(x_comp,y)]
print lrgs.best_score_
print lrgs.best_estimator_


# exp_var_score = [explained_variance_score(y[test],rf_reg.fit(x_comp[train],y[train]).predict(x_comp[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_comp,y)]
# mean_sq_error = [mean_squared_error(rf_reg.fit(x_comp[train],y[train]).predict(x_comp[test]),y[test]) for train, test in kf_total.split(x_comp,y)]
# model = [rf_reg.fit(x_comp[train],y[train]) for train,test in kf_total.split(x_comp,y)]
# print exp_var_score
# print mean_sq_error
# m = mean_sq_error.index(min(mean_sq_error))
# print model[m].get_params()
# sio.savemat('feat_imp_whole.mat',{'feat_imp_whole':lrgs.best_estimator_.feature_importances_})
print '______________________________'


print 'Rank Subtracted:'

lrgs2 = GridSearchCV(estimator=rf_reg2, param_grid=p_grid, scoring =my_scorer, n_jobs=-1)
lrgs2.fit(x_new,y)
print [explained_variance_score(y[test],lrgs2.fit(x_new[train],y[train]).predict(x_new[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_new,y)]
print [mean_squared_error(lrgs2.fit(x_new[train],y[train]).predict(x_new[test]),y[test]) for train, test in kf_total.split(x_new,y)]
print lrgs2.best_score_
print lrgs2.best_estimator_


# exp_var_score2 = [explained_variance_score(y[test],rf_reg2.fit(x_new[train],y[train]).predict(x_new[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_new,y)]
# mean_sq_error2 = [mean_squared_error(rf_reg2.fit(x_new[train],y[train]).predict(x_new[test]),y[test]) for train, test in kf_total.split(x_new,y)]
# model2 = [rf_reg2.fit(x_new[train],y[train]) for train,test in kf_total.split(x_new,y)]
# print exp_var_score2
# print mean_sq_error2
# m2 = mean_sq_error2.index(min(mean_sq_error2))
# print model2[m2].get_params()
# sio.savemat('feat_imp_sub.mat',{'feat_imp_sub':lrgs2.best_estimator_.feature_importances_})
print '______________________________'

print 'Reduced Rank:'
x_hat = pca.fit_transform(x)[:,39:]
lrgs3 = GridSearchCV(estimator=rf_reg3, param_grid=p_grid, scoring =my_scorer, n_jobs=-1)
lrgs3.fit(x_hat,y)
print [explained_variance_score(y[test],lrgs3.fit(x_hat[train],y[train]).predict(x_hat[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_hat,y)]
print [mean_squared_error(lrgs3.fit(x_hat[train],y[train]).predict(x_hat[test]),y[test]) for train, test in kf_total.split(x_hat,y)]
print lrgs3.best_score_
print lrgs3.best_estimator_

# exp_var_score3 = [explained_variance_score(y[test],rf_reg3.fit(x_hat[train],y[train]).predict(x_hat[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_hat,y)]
# mean_sq_error3 = [mean_squared_error(rf_reg3.fit(x_hat[train],y[train]).predict(x_hat[test]),y[test]) for train, test in kf_total.split(x_hat,y)]
# model3 = [rf_reg3.fit(x_hat[train],y[train]) for train,test in kf_total.split(x_hat,y)]
# print exp_var_score3
# print mean_sq_error3
# m3 = mean_sq_error3.index(min(mean_sq_error3))
# print model3[m3].get_params()
# sio.savemat('feat_imp_rr.mat',{'feat_imp_rr':lrgs3.best_estimator_.feature_importances_})
print '______________________________'

sPCA_MSE,sPCA_MSE_test,sPCA_exp,sPCA_exp_test,sPCA_r2,sPCA_r2_test = visualise_results(kf_total,x_comp,y,lrgs.best_estimator_,-1)
sio.savemat('Metrics_full_train.mat',{'sPCA_MSE':sPCA_MSE,'sPCA_MSE_test':sPCA_MSE_test,'sPCA_exp':sPCA_exp,'sPCA_exp_test':sPCA_exp_test,'sPCA_r2':sPCA_r2,'sPCA_r2_test':sPCA_r2_test})

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/grid_search/aut/'+ task + '/rank_subtracted/vis/'
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)
sPCA_MSE,sPCA_MSE_test,sPCA_exp,sPCA_exp_test,sPCA_r2,sPCA_r2_test = visualise_results(kf_total,x_new,y,lrgs2.best_estimator_,-1)
sio.savemat('Metrics_full_train.mat',{'sPCA_MSE':sPCA_MSE,'sPCA_MSE_test':sPCA_MSE_test,'sPCA_exp':sPCA_exp,'sPCA_exp_test':sPCA_exp_test,'sPCA_r2':sPCA_r2,'sPCA_r2_test':sPCA_r2_test})


newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/grid_search/aut/'+ task + '/reduced_rank/vis/'
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)
sPCA_MSE,sPCA_MSE_test,sPCA_exp,sPCA_exp_test,sPCA_r2,sPCA_r2_test = visualise_results(kf_total,x_hat,y,lrgs3.best_estimator_,-1)
sio.savemat('Metrics_full_train.mat',{'sPCA_MSE':sPCA_MSE,'sPCA_MSE_test':sPCA_MSE_test,'sPCA_exp':sPCA_exp,'sPCA_exp_test':sPCA_exp_test,'sPCA_r2':sPCA_r2,'sPCA_r2_test':sPCA_r2_test})

sys.stdout.close()
