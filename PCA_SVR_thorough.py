import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model,cross_validation,grid_search
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.svm import SVR

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('dataset_ADOSTotalScore.mat')
x= dataset['data']
y = dataset['y']
y = np.ravel(y)
print y.shape

kf_total = cross_validation.KFold(len(x), n_folds=10,shuffle=True, random_state=78)
sklearn_pca = sklearnPCA()
svr_rbf = SVR(kernel='poly',degree =3)
pca_svr = Pipeline([('pca',sklearn_pca), ('svr', svr_rbf)])

sPCA_MAE =[]
sPCA_r2 =[]
sPCA_exp =[]

sPCA_MAE_test =[]
sPCA_r2_test =[]
sPCA_exp_test =[]

c_range = np.logspace(-2, 2, 5)
n_comp = np.linspace(20,40,5)

my_scorer = make_scorer(explained_variance_score)
lrgs = grid_search.GridSearchCV(estimator=pca_svr, param_grid=dict(pca__n_components =n_comp,svr__C = c_range), scoring =my_scorer, n_jobs=1)
print [explained_variance_score(y[test],lrgs.fit(x[train],y[train]).predict(x[test]),multioutput='variance_weighted') for train, test in kf_total]
print [mean_absolute_error(lrgs.fit(x[train],y[train]).predict(x[test]),y[test]) for train, test in kf_total]
print lrgs.best_score_
print lrgs.best_estimator_
model = lrgs.best_estimator_

for train, test in kf_total:
     
    sPCA_MAE.append(mean_absolute_error(y[train], model.predict(x[train])))
    sPCA_r2.append(r2_score(y[train], model.predict(x[train]), multioutput='variance_weighted'))
    sPCA_exp.append(explained_variance_score(y[train], model.predict(x[train]), multioutput='variance_weighted'))
    print 'MAE : ', mean_squared_error(y[train], model.predict(x[train]))
    print 'Explained Variance Score : ', explained_variance_score(y[train], model.predict(x[train]))
    print 'r2 score: ' , r2_score(y[train], model.predict(x[train]))
    fig, ax = plt.subplots()
    ax.scatter(y[train],model.predict(x[train]),y[train])
    ax.plot([y[train].min(), y[train].max()], [y[train].min(), y[train].max()], 'k--', lw=4)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Measured')
    plt.show() 

    sPCA_MAE_test.append(mean_absolute_error(y[test], model.predict(x[test])))
    sPCA_r2_test.append(r2_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    sPCA_exp_test.append(explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    print 'MAE : ', mean_squared_error(y[test], model.predict(x[test]))
    print 'Explained Variance Score : ', explained_variance_score(y[test], model.predict(x[tr]))
    print 'r2 score: ' , r2_score(y[test], model.predict(x[test]))
    fig, ax = plt.subplots()
    ax.scatter(y[test],model.predict(x[test]),y[test])
    ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Measured')
    plt.show() 

print(np.mean(sPCA_MAE),np.mean(sPCA_r2),np.mean(sPCA_exp))
print(np.mean(sPCA_MAE_test),np.mean(sPCA_r2_test),np.mean(sPCA_exp_test))