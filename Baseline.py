from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io as sio
import os
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
os.chdir('/home/niharikashimona/Downloads/Datasets/')

dataset = sio.loadmat('Aut_classify.mat') 
data = dataset['data']
y = dataset['y']
print y.shape
train = data[0:100,:]
train_y = y[0:100]
test = data[101:,:]
test_y = y[101:]

print train.shape,train_y.shape
# lr = linear_model.LinearRegression()
lr = linear_model.LogisticRegression(C=1e5)
# predicted = cross_val_predict(lr, data, y, cv=5)
# scores = cross_val_score(lr, data, y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

clf = SVC(kernel ='rbf',gamma = 0.7)
clf.fit(train,train_y)
print (clf.predict(test)).T
print test_y.T
# scores = cross_val_score(clf, data, y, cv =10)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))