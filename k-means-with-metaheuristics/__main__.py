import configparser, random 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cluster import Clustering
from genetic import Genetic
from generation import Generation
from sklearn.metrics import davies_bouldin_score

NORMALIZATION = True


def readVars(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    budget = int(config.get("vars", "budget"))
    kmax = int(config.get("vars", "kmax"))  # Maximum number of Clusters
    numOfInd = int(config.get("vars", "numOfInd"))  # number of individual
    Ps = float(config.get("vars", "Ps")) # portion of individuals to be replaced
    Pm = float(config.get("vars", "Pm")) # probability of mutation 
    Pc = float(config.get("vars", "Pc")) # portion that participates in crossover
    t = int(config.get("vars", "t")) # tournament size
    Pe = int(config.get("vars", "Pe")) # number of elite individual

    return budget, kmax, Ps, Pm, Pc, t, Pe, numOfInd

# minmax normalization
def minmax(data):
    normData = data
    data = data.astype(float)
    normData = normData.astype(float)
    for i in range(0, data.shape[1]):
        tmp = data.iloc[:, i]
        # max of each column
        maxElement = np.amax(tmp)
        # min of each column
        minElement = np.amin(tmp)

        # norm_dat.shape[0] : size of row
        for j in range(0, normData.shape[0]):
            normData[i][j] = float(
                data[i][j] - minElement) / (maxElement - minElement)

    normData.to_csv('result/norm_data.csv', index=None, header=None)
    return normData



if __name__ == '__main__':
    random.seed(15)
    config_file = "config.txt"
    if(NORMALIZATION):
        data = pd.read_csv('data/wdbc.csv', header=None)
        data = minmax(data)  # normalize
    else:
        data = pd.read_csv('result/norm_data.csv', header=None)
    # size of column
    dim = data.shape[1]

    # kmeans parameters & GA parameters
    generationCount = 0
    budget, kmax, Ps, Pm, Pc, t, Pe, numOfInd = readVars(config_file)

    print("-------------GA Info-------------------")
    print("budget", budget) 
    print("kmax", kmax)
    print("numOfInd", numOfInd)
    print("Ps", Ps)
    print("Pm", Pm)
    print("Pc", Pc)
    print("t", t)
    print("Pe", Pe)
    print("---------------------------------------")
    
    sequences = []
    for i in range(numOfInd):
        a = data.sample(n=kmax)
        b = []
        for j in range(kmax):
            b.extend(list(a.iloc[j,:]))
        sequences.append(b)

    chromosome_length = kmax * dim
    initial = Generation(numOfInd, 0)
    initial.randomGenerate(chromosome_length)  # initial generate chromosome
    #initial.generateFromInput(sequences)
    clustering = Clustering(initial, data, kmax)  # eval fit of chromosomes
    generation = clustering.calcChromosomesFit() 
    best = generation.chromosomes[0]
    
    dbiList = []
    labels = []
   # intraList = []
    tolerance = 0
    # ------------------------GA----------------------#
    while generationCount <= budget:
        GA = Genetic(numOfInd, Ps, Pm, Pc, t, Pe, budget, data, generationCount, kmax)
        generation, generationCount = GA.geneticProcess(generation)
        iBest = generation.chromosomes[0]
        
        #intra = clustering.intraClusterDistance(iBest)
        labels, dbi = clustering.printIBest(iBest)
        #db_Index = davies_bouldin_score(data, labels)
        dbiList.append(dbi)
        print(dbi)
        #print("calculated: ", dbi, "\n skleran: ", db_Index)
        #intraList.append(intra)
        
        if iBest.fitness>best.fitness:
            best = iBest
            tolerance = 0
        elif iBest.fitness == best.fitness: 
            tolerance += 1
        if iBest.fitness < best.fitness:
            tolerance = 0
        if tolerance == 10:
            break

    # ------------------output result-------------------#
    #clustering.output_result(iBest, data)
    

    print(labels)
    #db_Index = davies_bouldin_score(data, labels)
    #print(db_Index)

    '''plt.plot(accuracyList)
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title(' '+'\n Max Accuracy = '+ str(max(accuracyList)))
    plt.show()'''
    
    plt.plot(dbiList)
    plt.xlabel('Generation')
    plt.ylabel('Davies-Bouldin Index')
    plt.title(' ' +'\n Lowest Davies-Bouldin Index = '+ str(min(dbiList)))
    plt.show()
    
    '''plt.plot(intraList)
    plt.xlabel('Generation')
    plt.ylabel('Avg. Intra Cluster Distances')
    plt.title(' ' +'\n Min Intra Cluster Distance = '+ str(min(intraList)))
    plt.show()'''
    
    
    