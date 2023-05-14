import random

def generateAPermutation(n,source):
    perm = [i for i in range(n)]
    pos1 = random.randint(0, n - 1)
    pos2 = random.randint(0, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    for i in range(n):
        if perm[i] == source:
            perm[0],perm[i] = perm[i],perm[0]
            break
    return perm


class Chromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.__repres = generateAPermutation(self.__problParam['noNodes'],self.__problParam['source'])
        self.__fitness = 0.0
    
    @property
    def repres(self):
        return self.__repres
    
    @property
    def fitness(self):
        return self.__fitness 
    
    @repres.setter
    def repres(self, l = []):
        self.__repres = l 
    
    @fitness.setter 
    def fitness(self, fit = 0.0):
        self.__fitness = fit 
    
    def crossover(self, c):
        k = random.randint(0, len(self.__repres) - 1)
        newRepres = []
        for i in range(k):
            newRepres.append(self.__repres[i])
        counter = len(self.__repres) - k
        for i in range(len(c.__repres)):
            if counter == 0:
                break
            if c.__repres[i] not in newRepres:
                newRepres.append(c.__repres[i])
                counter -= 1
        for i in range(len(newRepres)):
            if newRepres[i] == self.__problParam['source']:
                newRepres[0],newRepres[i] = newRepres[i],newRepres[0]
                break       
        offspring = Chromosome(c.__problParam)
        offspring.repres = newRepres
        return offspring

    def mutation(self):
        pos1 = random.randint(1, len(self.__repres) - 1)
        pos2 = random.randint(1, len(self.__repres) - 1)
        self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]

    def __str__(self):
        # return '\nChromo: ' + str(self.__repres) + \
        # ' has fit: ' + str(self.__fitness)
        return self.__problParam['printFunction']( \
            self.__repres, self.__fitness, \
            self.__problParam['source'], self.__problParam['destination'])
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness