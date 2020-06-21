# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:52:23 2020

@author: mugdh
"""

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

#Importing data file
#file = input("Enter data file\t")
file="P4Data.txt"
Data = np.loadtxt(file,skiprows = 1)

#Importing centroid file
#centroid = input("Enter data file\t")
centroid = "P4Centroids.txt"
Initial_centroids = np.loadtxt(centroid,skiprows = 1)
print("Initial centroids are:")
print(Initial_centroids)

#Euclidean distance
def Euclidean(p, c, ax=1):
    return (np.linalg.norm((p - c), axis=ax))

Centroid_old = np.zeros(Initial_centroids.shape)
clusters = np.zeros(len(Data))
e = Euclidean(Initial_centroids, Centroid_old, None)

#Plot the data with inital centroids
colors = ['r', 'g']
fig, ax = plt.subplots(figsize=(10,5))
plt.title("Initial data points") 
plt.xlabel("x axis") 
plt.ylabel("y axis") 
plt.scatter(Data[:,0],Data[:,1], c='Blue', s=20)
plt.scatter(Initial_centroids[:,0], Initial_centroids[:,1], marker='^', s=60, c=['r', 'g'])
plt.show()

#Kmeans and updating clusters
number_clusters = 2
while  e != 0:
    for i in range(len(Data)):
        distances = Euclidean(Data[i], Initial_centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    Centroid_old = deepcopy(Initial_centroids)
    for i in range(number_clusters):
        points = [Data[j] for j in range(len(Data)) if clusters[j] == i]
        Initial_centroids[i] = np.mean(points, axis=0)
    e = Euclidean(Initial_centroids, Centroid_old, None)

#Plotting the data with final centroids    
colors = ['r', 'g']
fig, ax = plt.subplots(figsize=(10,5))
for i in range(number_clusters):
    plt.title("Clustered data points") 
    plt.xlabel("x axis") 
    plt.ylabel("y axis") 
    points = np.array([Data[j] for j in range(len(Data)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=20, c=colors[i])
ax.scatter(Initial_centroids[:, 0], Initial_centroids[:, 1], marker='^', s=60, c=['r', 'g'])
plt.show()

#Printing final centroids
print("Final centroids are:")
print(Centroid_old)

#J  value calculation
a = Centroid_old[0][0]
b = Centroid_old[0][1]
c = Centroid_old[1][0]
d = Centroid_old[1][1]

def Cost_function(p, c):
    return (pow(p[0]-c[0],2) + pow(p[1]-c[1],2))
j = 0
for i in range(0,len(clusters)):
    if(clusters[i] == 0):
        points = (Data[i,0],Data[i,1])
        centre = (a,b)
        j+=Cost_function(points,centre)
    else:
        points = (Data[i,0],Data[i,1])
        centre = (c,d)
        j+=Cost_function(points,centre)
j=j/len(Data)      

print("\nValue of J is:",j)


