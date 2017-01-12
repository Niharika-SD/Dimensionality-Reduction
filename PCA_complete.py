import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pylab import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import KernelPCA as sklearnKPCA
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('bytSrsP2SocCommInteractionRaw.mat')
data = dataset['data']
y = dataset['y']
print data
# [r,c] = data.shape
# split = np.ceil(r*0.8)
# data_train = data[0:split,:]
# y_train = y[0:split]
# data_test = data[split+1:,:]
# y_test = y[split+1:]


sklearn_pca = sklearnPCA(n_components=5)
sklearn_transf = sklearn_pca.fit_transform(data)
print(sklearn_pca.explained_variance_ratio_)
print(sklearn_transf.shape)
# red_data_test = sklearn_pca.transform(data_test)

regr = linear_model.LinearRegression()

# regr.fit(sklearn_transf,y)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(red_data_test)-y_test)**2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(red_data_test,y_test))
predicted = cross_val_predict(regr,sklearn_transf ,y, cv=10)
scores = cross_val_score(regr, sklearn_transf , y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



sklearn_kpca = sklearnKPCA(n_components=10	,kernel="rbf")
sklearn_transf_kpca = sklearn_kpca.fit_transform(data)
# red_data_test = sklearn_kpca.transform(data_test)

regr = linear_model.LinearRegression()

# regr.fit(sklearn_transf,y_train)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((regr.predict(red_data_test)-y_test)**2))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(red_data_test,y_test))
sredicted = cross_val_predict(regr,sklearn_transf_kpca,y, cv=10)
scores = cross_val_score(regr, sklearn_transf_kpca, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()