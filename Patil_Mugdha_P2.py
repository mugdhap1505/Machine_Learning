# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:24:21 2020

@author: mugdha
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 

def WeightInitialization(nfeatures):
	w = np.zeros([nfeatures,1])
	return w

def SigmoidFunction(X,w):
	return 1/(1 + np.exp(-np.dot(X,w)))

def Add_intercept(x):
    intercept = np.ones((x.shape[0], 1))
    return np.concatenate((intercept, x), axis=1) 

def ModelEstimation(X,y,w,m,alpha,n,mT):
	costs = []
	for i in range(n):
		H = SigmoidFunction(X,w)
		cost = -y * np.log(H) - (1 - y) * np.log(1 - H)
		j = (1/len(m)) * np.dot(mT,cost)
		costs.append(j)
		old_w = np.dot(np.subtract(H,y).T,X)
		w = w - alpha * old_w.T
	return w,H,costs,j,cost
	
#Import Dataset
Dataset = "DivorceAll.txt"
data = np.loadtxt(Dataset,skiprows = 1)
np.take(data,np.random.permutation(data.shape[0]),axis=0, out = data)

#Splitting into Training and Test Data
train = data[0:136,]
test = data[136:,]

#Saving the Training and Test Data files
np.savetxt('Patil_Mugdha_Train.txt', train, fmt="%0.2f",delimiter='\t',header = "136\t54",comments = "")
np.savetxt('Patil_Mugdha_Test.txt', test, fmt="%0.2f",delimiter='\t',header = "34\t54",comments = "")

# Training Data
Training_File = 'Patil_Mugdha_Train.txt'
X = np.loadtxt(Training_File ,skiprows = 1)
m = np.ones([len(X),1])
y = X[:,-1]
y = np.reshape(y, (136, 1))
X = X[:,:-1]
X = np.concatenate((m,X),axis = 1)
w = WeightInitialization(X.shape[1])
alpha = 0.0001
n = 1000
mT = np.ones([1,X.shape[0]])

#Applying the model
w,H,costs,j,cost = ModelEstimation(X,y,w,m,alpha,n,mT)

#Plotting iterations vs J for training data
costs = np.reshape(costs,(1000,1))
iterations = list(range(1,len(costs)+1))
plt.title("Cost Reduction")
plt.xlabel("Iterations")
plt.ylabel("J")
plt.plot(iterations,costs,'r')

#Testing Data 
Test_File = 'Patil_Mugdha_Test.txt'
X_test = np.loadtxt(Test_File,skiprows = 1)
m_test = np.ones([len(X_test),1])
y_test = X_test[:,-1]
y_test = np.reshape(y_test, (34, 1))
X_test = X_test[:,:-1]
mT_test = np.ones([len(X_test),1])
X_test = np.concatenate([mT_test,X_test],axis = 1)
H_test = SigmoidFunction(X_test,w)
mT_test = np.ones([1,X_test.shape[0]])

cost_test = -y_test * np.log(H_test) - (1-y_test) * np.log(1-H_test)
J_test = (1/len(m_test)) * np.dot(mT_test,cost_test)
y_pred = np.round(H_test)

#Final J value
print("Final J is",J_test)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\nTrue Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#Accuracy
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("\nAccuracy {:0.2f}%".format(Accuracy))

#Precision 
Precision = tp/(tp+fp) 
print("\nPrecision {:0.2f}".format(Precision))

#Recall
Recall = tp/(tp+fn) 
print("\nRecall {:0.2f}".format(Recall))

#F1 Score
f1 = (2*Precision*Recall)/(Precision + Recall)
print("\nF1 Score {:0.2f}".format(f1))
print("\nReport",classification_report(y_test, y_pred))

#Plotting iteraions vs J for Test Data
from matplotlib import pyplot as plt
iterations = list(range(1,len(costs)+1))
plt.title("Cost Reduction")
plt.xlabel("Iterations")
plt.ylabel("J")
plt.plot(iterations,costs,'c')

