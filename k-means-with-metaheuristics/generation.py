from chromosome import Chromosome
import pandas as pd

class Generation:
    def __init__(self, numberOfIndividual, generationCount):
        self.numberOfIndividual = numberOfIndividual
        self.chromosomes = []
        self.generationCount = generationCount

    def sortChromosomes(self):
        self.chromosomes = sorted(self.chromosomes, reverse=True, key=lambda elem: elem.fitness)
        return self.chromosomes

    def randomGenerate(self, lengthOfChromosome):
        for i in range(0, self.numberOfIndividual): 
            chromosome = Chromosome([], lengthOfChromosome)
            chromosome.randomGenerateChromosome()
            self.chromosomes.append(chromosome)
    
    def generateFromInput(self, geneSequences):
        if len(geneSequences) == self.numberOfIndividual:
            for i in range(0, self.numberOfIndividual):
                length = len(geneSequences[i])
                chromosome = Chromosome(geneSequences[i], length)
                self.chromosomes.append(chromosome)
        else:
            print(" Number of genes given is not equal to the number of individuals in this generation! ")