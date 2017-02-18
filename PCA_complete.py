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

kf_total = cross_validation.KFold(len(x), n_folds=10,shuffle=True, random_state=782828)
sklearn_pca = sklearnPCA(n_components=30)
sklearn_kpca = sklearnKPCA(n_components=25,kernel="rbf",fit_inverse_transform = 'True')
svr_rbf = SVR(kernel='poly',degree =3)
lr = linear_model.LinearRegression()

pca_lr = Pipeline([('pca',sklearn_pca), ('lr', lr)])
kpca_lr = Pipeline([('kpca',sklearn_kpca), ('lr', lr)])
kpca_svr = Pipeline([('kpca',sklearn_kpca), ('svr', svr_rbf)])
pca_svr = Pipeline([('pca',sklearn_pca), ('svr', svr_rbf)])


print " accuracy on reduced dataset using PCA \n"
PCA_MAE =[]
PCA_r2 =[]
PCA_exp =[]

for train, test in kf_total:

    model = pca_svr.fit(x[train],y[train])
    # print y[test], model.predict(x[test])
    print 'MAE : ', mean_absolute_error(y[test], model.predict(x[test]))
    PCA_MAE.append(mean_absolute_error(y[test], model.predict(x[test])))
    # print 'r2 : ', r2_score(y[test], model.predict(x[test]))
    PCA_r2.append(r2_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    print 'explained variance score', explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted')
    PCA_exp.append(explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    fig, ax = plt.subplots()
    ax.scatter(y[test],model.predict(x[test]),y[test])
    ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

print(np.mean(PCA_MAE),np.mean(PCA_r2),np.mean(PCA_exp))

print " Baseline \n"

MAE =[]
r2 =[]
exp_v =[]
for train, test in kf_total:

    model = lr.fit(x[train],y[train])
    print y[test], model.predict(x[test])
    print 'MAE : ', mean_absolute_error(y[test], model.predict(x[test]))
    print 'r2 : ', r2_score(y[test], model.predict(x[test]))
    print 'explained variance score', explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted')
    MAE.append(mean_absolute_error(y[test], model.predict(x[test])))
    r2.append(r2_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    exp_v.append(explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    fig, ax = plt.subplots()
    ax.scatter(y[test],model.predict(x[test]),y[test])
    ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

print(np.mean(MAE),np.mean(r2),np.mean(exp_v))

print " accuracy on reduced dataset using kPCA \n"
kPCA_MAE =[]
kPCA_r2 =[]
kPCA_exp =[]
for train, test in kf_total:

    model = kpca_svr.fit(x[train],y[train])
    # print y[test], model.predict(x[test])
    print 'MAE : ', mean_absolute_error(y[test], model.predict(x[test]))
    # print 'r2 : ', r2_score(y[test], model.predict(x[test]))
    print 'explained variance score', explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted')
    kPCA_MAE.append(mean_absolute_error(y[test], model.predict(x[test])))
    kPCA_r2.append(r2_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    kPCA_exp.append(explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted'))
    # print()
    fig, ax = plt.subplots()
    ax.scatter(y[test],model.predict(x[test]),y[test])
    ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


print(np.mean(kPCA_MAE),np.mean(kPCA_r2),np.mean(kPCA_exp))
