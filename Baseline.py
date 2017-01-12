from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score

os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('bytSrsP2SocCommInteractionRaw.mat') 
data = dataset['data']
y = dataset['y']
print dataset

lr = linear_model.LinearRegression()
predicted = cross_val_predict(lr, data, y, cv=10)
scores = cross_val_score(lr, data, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()