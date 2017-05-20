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
from SparsePCA_Pipeline import Split_class,create_dataset,evaluate_results

df_aut,df_cont = Split_class()
task  ='ADOS.Total'
x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data')	
	
# x =np.concatenate((x_cont,x_aut),axis =0)
# y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))
x = x_aut
y= np.ravel(y_aut)

pca =PCA()
pca1 = PCA(n_components = 38)

x_comp = x 
x_new = pca.inverse_transform(pca.fit_transform(x))-pca1.inverse_transform(pca1.fit_transform(x)) 
svr_poly = NuSVR(kernel ='rbf',cache_size=1000)
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg2 = RandomForestRegressor(n_estimators=100)
rf_reg3 = RandomForestRegressor(n_estimators=100)

kf_total = KFold(n_splits=10,shuffle=True, random_state=3)

ridge_range = np.linspace(0.01,0.09,5)
c_range = np.logspace(0,5,5)
gamma_range = np.logspace(-4, 1, 6)
my_scorer = make_scorer(explained_variance_score)
p_grid = dict(C =c_range,nu=ridge_range, gamma = gamma_range)

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/aut/'+ task + '/full/'
if not os.path.exists(newpath):
	os.makedirs(newpath)
os.chdir(newpath)

# lrgs = GridSearchCV(estimator=svr_poly, param_grid=p_grid, scoring =my_scorer, n_jobs=1)
# lrgs.fit(x,y)
sys.stdout=open('results'+'.txt',"w")
# print [explained_variance_score(y[test],lrgs.fit(x[train],y[train]).predict(x[test]),multioutput='variance_weighted') for train, test in kf_total.split(x,y)]
# print [mean_absolute_error(lrgs.fit(x[train],y[train]).predict(x[test]),y[test]) for train, test in kf_total.split(x,y)]
# print lrgs.best_score_
# print lrgs.best_estimator_

print 'Original :'
exp_var_score = [explained_variance_score(y[test],rf_reg.fit(x_comp[train],y[train]).predict(x_comp[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_comp,y)]
mean_sq_error = [mean_squared_error(rf_reg.fit(x_comp[train],y[train]).predict(x_comp[test]),y[test]) for train, test in kf_total.split(x_comp,y)]
model = [rf_reg.fit(x_comp[train],y[train]) for train,test in kf_total.split(x_comp,y)]
print exp_var_score
print mean_sq_error
m = mean_sq_error.index(min(mean_sq_error))
print model[m].get_params()
sio.savemat('feat_imp_whole.mat',{'feat_imp_whole':model[m].feature_importances_})
print '______________________________'


print 'Rank Subtracted:'
exp_var_score2 = [explained_variance_score(y[test],rf_reg2.fit(x_new[train],y[train]).predict(x_new[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_new,y)]
mean_sq_error2 = [mean_squared_error(rf_reg2.fit(x_new[train],y[train]).predict(x_new[test]),y[test]) for train, test in kf_total.split(x_new,y)]
model2 = [rf_reg2.fit(x_new[train],y[train]) for train,test in kf_total.split(x_new,y)]
print exp_var_score2
print mean_sq_error2
m2 = mean_sq_error2.index(min(mean_sq_error2))
print model2[m2].get_params()
sio.savemat('feat_imp_sub.mat',{'feat_imp_sub':model2[m2].feature_importances_})
print '______________________________'

print 'Reduced Rank:'
x_hat = pca.fit_transform(x)[:,39:]
exp_var_score3 = [explained_variance_score(y[test],rf_reg3.fit(x_hat[train],y[train]).predict(x_hat[test]),multioutput='variance_weighted') for train, test in kf_total.split(x_hat,y)]
mean_sq_error3 = [mean_squared_error(rf_reg3.fit(x_hat[train],y[train]).predict(x_hat[test]),y[test]) for train, test in kf_total.split(x_hat,y)]
model3 = [rf_reg3.fit(x_hat[train],y[train]) for train,test in kf_total.split(x_hat,y)]
print exp_var_score3
print mean_sq_error3
m3 = mean_sq_error3.index(min(mean_sq_error3))
print model3[m3].get_params()
sio.savemat('feat_imp_rr.mat',{'feat_imp_rr':model3[m3].feature_importances_})
print '______________________________'

evaluate_results(kf_total,x_comp,y,model[m],-1)
os.makedirs(r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/aut/'+ task + '/rank_subtracted/')
os.chdir(r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/aut/'+ task + '/rank_subtracted/')
evaluate_results(kf_total,x_new,y,model2[m2],-1)
os.makedirs(r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/aut/'+ task + '/reduced_rank/')
os.chdir(r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/PCA_l2/rf_reg/aut/'+ task + '/reduced_rank/')
evaluate_results(kf_total,x_hat,y,model3[m3],-1)

sys.stdout.close()
