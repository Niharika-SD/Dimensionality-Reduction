from sklearn.model_selection import cross_val_predict
from sklearn import linear_model,cross_validation,grid_search
import scipy.io as sio
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score


os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('dataset_ADOSTotalScore.mat')
x= dataset['data']
y = dataset['y']
y = np.ravel(y)
print y.shape
kf_total = cross_validation.KFold(len(x), n_folds=10, shuffle=True, random_state=782828)
lr = linear_model.LinearRegression()
model = PLSRegression(n_components=30)
sklearn_pca = sklearnPCA(n_components=5)
pca_lr = Pipeline([('pca',sklearn_pca), ('lr', lr)])

for train, test in kf_total:
	
	model.fit(x[train], y[train])
	print y[test], model.predict(x[test])
	print 'MAE : ', mean_absolute_error(y[test], model.predict(x[test]))
	print 'explained variance score', explained_variance_score(y[test], model.predict(x[test]), multioutput='variance_weighted')
	print(model.coef_)
	# fig, ax = plt.subplots()
	# ax.scatter(y[test],model.predict(x[test]),y[test])
	# ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
	# ax.set_xlabel('Predicted')
	# ax.set_ylabel('Measured')
	# plt.show() 
