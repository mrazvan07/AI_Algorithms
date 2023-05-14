

from Chromosome import Chromosome, generateAPermutation
from GA import GA,computeTotalWeightOfPath,readNet


gaParam = {'popSize' : 100, 'noGen':1000}
param = readNet("data\medium.txt")
problParam = {'function' : computeTotalWeightOfPath, 'noNodes': param['noNodes'], 'source': param['source']}

ga = GA(gaParam,problParam)
ga.initialisation()
ga.evaluation()

bestOfBest = Chromosome(problParam)
bestOfBest.fitness = 1000000

for generation in range(gaParam['noGen']):
    ga.oneGenerationElitism()
    bestChr = ga.bestChromosome()
    if bestOfBest.fitness > bestChr.fitness:
        bestOfBest = bestChr

print('Best solution: ' + str(bestOfBest.repres))
print('Best Fitness: ' + str(bestOfBest.fitness))

for gen in ga.last_generation():
    if gen.fitness == bestOfBest.fitness:
        print(gen.repres)
