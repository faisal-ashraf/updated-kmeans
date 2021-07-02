import pandas as pd
import math
import json
import numpy as np
from sklearn.metrics import davies_bouldin_score

pd.options.mode.chained_assignment = None

class Point:
    def __init__(self, pattern_id):
        self.pattern_id = pattern_id
        self.length = len(pattern_id)
        self.z = -1

    def __str__(self):
        return str(self.pattern_id)

    def toJSON(self):
        return {
            'pattern_id':self.pattern_id
        }


class Cluster:
    def __init__(self, data_dimension, centroid):
        self.dim = data_dimension
        self.centroid = centroid
        self.points = []
        self.distances = []

    # this method finds the average distance of all elements in cluster to its centroid
    def computeS(self):
        n = len(self.points)
        if n == 0:
            return 0
        s = 0
        for x in self.distances:
            s += x
        return float(s / n)


class Clustering:
    def __init__(self, generation, data, kmax):
        self.generation = generation
        self.data = data
        self.dim = data.shape[1]
        self.penalty = 1000000
        self.kmax = kmax
    
    
    
    def daviesBouldin(self, clusters):
        sigmaR = 0.0
        nc = len(clusters)
        for i in range(nc):
            sigmaR = sigmaR + self.computeR(clusters)
            #print(sigmaR)
        DBIndex = float(sigmaR) / float(nc)
        return DBIndex

    def computeR(self, clusters):
        listR = []
        for i, iCluster in enumerate(clusters):
            for j, jCluster in enumerate(clusters):
                if(i != j):
                    temp = self.computeRij(iCluster, jCluster)
                    listR.append(temp)
        return max(listR)

    def computeRij(self, iCluster, jCluster):
        Rij = 0
        d = self.euclidianDistance(iCluster.centroid, jCluster.centroid)
        #print("d",d)
        #print("icluster",iCluster.computeS())
        Rij = (iCluster.computeS() + jCluster.computeS()) / d
        #print("Rij:", Rij)
        return Rij


    def euclidianDistance(self, point1, point2):
        sum = 0
        for i in range(0, point1.length):
            square = pow(point1.pattern_id[i] - point2.pattern_id[i], 2)
            sum += square
        sqr = math.sqrt(sum)
        return sqr

    def calcDistance(self, clusters):
        kmax = self.kmax
        dim = self.dim
        data = self.data
        dis = 0
        disSet = []

        for z in range(data.shape[0]):
            point = Point(data.loc[z][0:dim])
            point.z = z
            for i in range(kmax):
                dis = self.euclidianDistance(clusters[i].centroid, point)
                disSet.append(dis)
                dis = 0
            n = disSet.index(min(disSet))  # n is the cluster number with minimum distance
            minDis = disSet[n]
            clusters[n].points.append(point)
            clusters[n].distances.append(minDis)
            disSet = []  # clear disSet	# calculate distance
        return clusters



    def intraClusterDistance(self, chromosome):
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(chromosome.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))
        clusters = self.calcDistance(clusters)
        summ = 0 
        for c in clusters:
            summ += c.computeS()
        return float(summ/len(clusters))



    # childChromosome, kmax
    def calcChildFit(self, childChromosome):
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(childChromosome.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))
        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)
        #DBIndex = self.intraClusterDistance(clusters)
        childChromosome.fitness = 1 / DBIndex
        return childChromosome
    
    
    
    def calcChromosomesFit(self): # find fitness of all the individuals
        kmax = self.kmax
        generation = self.generation
        numOfInd = generation.numberOfIndividual
        chromo = generation.chromosomes
        for i in range(0, numOfInd):
            dim = self.dim
            clusters = []
            for j in range(kmax):
                point = Point(chromo[i].genes[j * dim: (j + 1) * dim])
                clusters.append(Cluster(dim, point))
            clusters = self.calcDistance(clusters)
            DBIndex = self.daviesBouldin(clusters)
            #DBIndex = self.intraClusterDistance(clusters)
            generation.chromosomes[i].fitness = 1 / DBIndex
        return generation



    def printIBest(self, iBest): # iBest is the best individual 
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))
        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)
        
        z = (np.zeros(self.data.shape[0])).tolist()
        for i, cluster in enumerate(clusters):
            for j in cluster.points:
                z[j.z] = i
        #print(z)
        #print("iBest Davies-Bouldin Index:", DBIndex)
        return z, DBIndex
        
    
    
    
    
    