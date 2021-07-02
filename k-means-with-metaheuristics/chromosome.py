import random

class Chromosome:
    def __init__(self, genes, length):
        self.genes = genes
        self.length = length
        self.fitness = 0

    def randomGenerateChromosome(self):
        for i in range(0, self.length, +1):
            gen = float('%.2f' % random.uniform(0.0, 1.0)) 
            self.genes.append(gen)
        return self 
    
    
    
    
    '''
    def randomGenerateChromosomeFromData(self, data, kmax):
        df = data.sample(n=kmax)
        #print(df)
        out = []
        for i in range(len(df)):
            a = list(df.iloc[i])
            out.extend(a)
        out = [ float('%.2f' % x) for x in out ]      
        self.genes = out
        return self '''
