# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:22:32 2021

@author: USER
"""

import pandas as pd
import random, math
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import euclidean
from cluster import Cluster, Point



def makeClusters(data,labels,k,centroids):
    dim = data.shape[1]
    clusters = []
    centers = []
    
    for i in range(k):
        p1 = Point(centroids[i])
        cl = Cluster(dim, p1)
        clusters.append(cl)
        #print(p1.length)
        centers.append(p1)
    
    for i in range(len(data)):
        p1 = Point(list(data.iloc[i,:]))
        clusters[labels[i]].points.append(p1)
       # print(p1.length)
       # print(labels[i])
        dis = euclidianDistance(p1, centers[labels[i]])
        clusters[labels[i]].distances.append(dis)
    return clusters
  
      
def daviesBouldin( clusters):
    sigmaR = 0.0
    nc = len(clusters)
    for i in range(nc):
        sigmaR = sigmaR + computeR(clusters)
    DBIndex = float(sigmaR) / float(nc)
    return DBIndex

def computeR( clusters):
    listR = []
    for i, iCluster in enumerate(clusters):
        for j, jCluster in enumerate(clusters):
            if(i != j):
                temp = computeRij(iCluster, jCluster)
                listR.append(temp)
    return max(listR)    

def computeRij( iCluster, jCluster):
    Rij = 0
    d = euclidianDistance(iCluster.centroid, jCluster.centroid)
    Rij = (iCluster.computeS() + jCluster.computeS()) / d
    return Rij

def euclidianDistance( point1, point2):
        sum = 0
        for i in range(0, point1.length):
            square = pow(point1.pattern_id[i] - point2.pattern_id[i], 2)
            sum += square
        sqr = math.sqrt(sum)
        return sqr       
    


kmax = 2 

data = pd.read_csv('result/norm_data.csv', header=None)

print("======= Kmeans =======")
kmeans = KMeans(n_clusters=kmax, random_state=1).fit(data)
labels = kmeans.labels_ 
centroids = kmeans.cluster_centers_
#db_Index = davies_bouldin_score(data, labels)

print(labels)
#print(db_Index)
print(centroids)

clsters = makeClusters(data, labels, kmax,centroids.tolist())
dbi = daviesBouldin(clsters)
print(dbi)


print("======= Minibatch Kmeans =======")
from sklearn.cluster import MiniBatchKMeans
kmeans2 = MiniBatchKMeans(n_clusters=kmax, random_state=0,batch_size=6).fit(data)
labels = kmeans2.labels_ 
centroids = kmeans2.cluster_centers_
#db_Index = davies_bouldin_score(data, labels)

print(labels)
#print(db_Index)
print(centroids)
clsters = makeClusters(data, labels, kmax,centroids.tolist())
dbi = daviesBouldin(clsters)
print(dbi)

