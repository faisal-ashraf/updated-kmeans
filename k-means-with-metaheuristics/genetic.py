import random
import numpy as np
from generation import Generation
from chromosome import Chromosome
from cluster import Clustering

random.seed(1)


class Genetic:
    def __init__(self, numberOfIndividual, Ps, Pm, Pc,t, Pe, budget, data, generationCount, kmax):
        self.numberOfIndividual = numberOfIndividual
        self.Ps = Ps
        self.Pm = Pm
        self.Pc = Pc
        self.t = t
        self.Pe = Pe
        self.budget = budget
        self.data = data
        self.generationCount = generationCount
        self.kmax = kmax

    def geneticProcess(self, generation):
        print("------------ Generation:", self.generationCount, " -------------")
        generation.sortChromosomes()
        generation2 = self.selection_TS(generation) #  Selection 
        #generation2 = self.crossover(generation2) #  Crossover 
        generation2 = self.mutation(generation2) #  Mutation

        j = 0
        for i in range(self.Pe-1, self.numberOfIndividual):
            generation.chromosomes[i] = generation2.chromosomes[j]
            j += 1
        self.generationCount += 1
        return generation, self.generationCount
    
    '''
    def hillClimb(self, generation):
        numOfInd = self.numberOfIndividual
        h = 5
        for i in range(numOfInd):
            best = generation.chromosomes[i]
            S = best.genes
            bestF = best.fitness
            for j in range(0, h):
                S = self.tweak(S)
                newS = Chromosome(S, len(S))
                clustering = Clustering(generation, self.data, self.kmax)
                newS = clustering.calcChildFit(newS)
                if newS.fitness > bestF:
                    bestF = newS.fitness
                    best = newS
            generation.chromosomes[i] = best 
        return generation
    
    def tweak(self, genes):
        l = len(genes)
        for i in range(l):
            genes[i] = float('%.2f' % random.uniform(0.0, 1.0))
        return genes
    '''

    def selection_ranking(self, generation):
        numOfInd = self.numberOfIndividual
        Ps = self.Ps
        # replace the worst Ps*numOfInd individual with the best Ps*numOfInd individual
        for i in range(0, int(Ps * numOfInd)):
            generation.chromosomes[numOfInd - 1 - i] = generation.chromosomes[i]
        #generation = self.hillClimb(generation)
        # sort chromosomes after ranking selection
        generation.sortChromosomes()
        return generation 
    
    def selection_TS(self, generation):
        numOfInd = self.numberOfIndividual
        Ps = self.Ps
        t = self.t
        #generation = self.hillClimb(generation)
        for i in range(0, int(Ps * numOfInd)):
            randomIdx = random.sample(range(0,numOfInd), t)
            t_players = [generation.chromosomes[k] for k in randomIdx]
            temp = self.runTournament(t_players)
            generation.chromosomes[numOfInd-1-i] = temp      
        # sort chromosomes after tournament selection
        generation.sortChromosomes()
        return generation
    
    def runTournament(self, chromosomes):
        bestFitness = 0
        bestChromosome = chromosomes[0]
        for ch in chromosomes:
            if ch.fitness > bestFitness:
                bestFitness = ch.fitness
                bestChromosome = ch
        return bestChromosome
    

    def crossover(self, generation):
        numOfInd = self.numberOfIndividual
        Pc = self.Pc
        x = int(Pc * (numOfInd-self.Pe))
        #print(numOfInd, x)
        index = random.sample(range(self.Pe, numOfInd - 1), x)
        #index = random.sample(range(0, numOfInd - 1), int(Pc * numOfInd))
        for i in range(int(len(index) / 2),+2):  # do how many time
            generation = self.doCrossover(generation, i, index)
        generation.sortChromosomes()
        return generation

    def doCrossover(self, generation, i, index):
        kmax = self.kmax
        dim = self.data.shape[1]
        chromo = generation.chromosomes
        #length = chromo[0].length
        #cut = random.randint(1, length - 1)
        #cuts = sorted(random.sample(range(1, 5), kmax-1))
        cuts = [a*dim for a in  range(1, kmax)]
        parent1 = chromo[index[i]]
        parent2 = chromo[index[i + 1]]
        child1gene = parent1.genes
        child2gene = parent2.genes
        if len(cuts)%2 == 0:
            p = float('%.2f' % random.uniform(0.0, 1.0))
            
            for i in range(0,len(cuts),2):
                if p > 0.5:
                    child1gene[cuts[i]:cuts[i+1]] =  parent2.genes[cuts[i]:cuts[i+1]]
                    child2gene[cuts[i]:cuts[i+1]] =  parent1.genes[cuts[i]:cuts[i+1]]
                else:
                    child1gene[cuts[i]:cuts[i+1]] =  parent1.genes[cuts[i]:cuts[i+1]]
                    child2gene[cuts[i]:cuts[i+1]] =  parent2.genes[cuts[i]:cuts[i+1]]
        elif len(cuts)%2 == 1:
            p = float('%.2f' % random.uniform(0.0, 1.0))
            
            child1gene[0:cuts[0]] =  parent2.genes[0:cuts[0]]
            child2gene[0:cuts[0]] =  parent1.genes[0:cuts[0]]
            for i in range(1, len(cuts), 2):
                if p > 0.5:
                    child1gene[cuts[i]:cuts[i+1]] =  parent2.genes[cuts[i]:cuts[i+1]]
                    child2gene[cuts[i]:cuts[i+1]] =  parent1.genes[cuts[i]:cuts[i+1]]
                else: 
                    child1gene[cuts[i]:cuts[i+1]] =  parent1.genes[cuts[i]:cuts[i+1]]
                    child2gene[cuts[i]:cuts[i+1]] =  parent2.genes[cuts[i]:cuts[i+1]]
        
        #genesChild1 = parent1.genes[0:cut] + parent2.genes[cut:length]
        #genesChild2 = parent1.genes[cut:length] + parent2.genes[0:cut]
        child1 = Chromosome(child1gene, len(child1gene))
        child2 = Chromosome(child2gene, len(child2gene))
        # ----clustering----
        clustering = Clustering(generation, self.data, self.kmax)
        child1 = clustering.calcChildFit(child1)
        child2 = clustering.calcChildFit(child2)
        # -------------------
        listA = []
        listA.append(parent1)
        listA.append(parent2)
        listA.append(child1)
        listA.append(child2)
        # sort parent and child by fitness / dec
        listA = sorted(listA, reverse=True,  key=lambda elem: elem.fitness)
        generation.chromosomes[index[i]] = listA[0]
        generation.chromosomes[index[i + 1]] = listA[1]
        return generation

    def mutation(self, generation):
        numOfInd = self.numberOfIndividual
        fitnessList = []
        generationAfterM = Generation(numOfInd, generation.generationCount)
        flagMutation = (np.zeros(numOfInd)).tolist()

        for i in range(numOfInd):
            temp = generation.chromosomes[i]
            fitnessList.append(temp.fitness)

        for i in range(numOfInd):
            if i < self.Pe:  # Ibest doesn't need mutation
                generationAfterM.chromosomes.append(generation.chromosomes[0])
                flagMutation[0] = 0
            else:
                generationAfterM = self.doMutation(generation.chromosomes[i],	generationAfterM, flagMutation, fitnessList, i)

        generationAfterM.sortChromosomes()
        return generationAfterM

    def doMutation(self, chromosomeBeforeM, generationAfterM, flagMutation, fitnessList, i):
        Pm = self.Pm
        dice = []
        length = len(chromosomeBeforeM.genes)
        chromosome = Chromosome([], length)
        geneFlag = []

        for j in range(length):
            dice.append(float('%.2f' % random.uniform(0.0, 1.0)))
            if dice[j] > Pm:
                chromosome.genes.append(chromosomeBeforeM.genes[j])
                geneFlag.append(0)

            if dice[j] <= Pm:
                chromosome.genes.append(float('%.2f' % random.uniform(0.0, 1.0)))
                geneFlag.append(1)

        check = sum(geneFlag)

        if check == 0:
            flagMutation[i] = 0
            chromosome.fitness = fitnessList[i]
        else:
            flagMutation[i] = 1

            #---clustering----
            clustering = Clustering(chromosomeBeforeM, self.data, self.kmax)
            chromosome = clustering.calcChildFit(chromosome)
            #------------------

        generationAfterM.chromosomes.append(chromosome)
        return generationAfterM
