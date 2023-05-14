import random
import math
from Ant import Ant

class ACO:
    def __init__(self, params = None, problParam = None):
        self.__params = params 
        self.__problParam = problParam #popSize, noGenerations, dynamic
        self.__population = [] 
        self.__bestSolution = []
        self.__bestDistance = 0
        
        
    @property
    def population(self):
        return self.__population
    
    def initialisation(self):
        for _ in range(self.__problParam['popSize']):
            a = Ant(self.__params)
            self.__population.append(a)

    def initialisationPheromones(self):
        for _ in range(self.__params['noNodes']):
            initial_pheromone_line = [1] * self.__params['noNodes']
            self.__params['pheromoneMat'].append(initial_pheromone_line)
            
    
    def runAco(self):
        iteration = 0
        while(iteration != self.__params['noNodes']):
            for ant in self.__population:
                added = self.chooseNextNode(ant)
                if added:
                    self.updateLocalPheromone(ant)
            iteration += 1
            if(self.bestAnt().distance() != 99999):
                self.updateGlobalPheromone(self.bestAnt())

    def get_bestSolution(self):
        return self.__bestSolution,self.__bestDistance

    def reinitializeAnts(self):
        for ant in self.__population:
            ant.reinitialize()

    def updateGlobalPheromone(self, ant):
        path = ant.path()
        for i in range(len(path) - 1):
            x = path[i]
            y = path[i + 1]
            self.__params['pheromoneMat'][x][y] = self.__params['evaporationRate'] * self.__params['pheromoneMat'][x][y] + self.__params['evaporationRate'] * (1/ant.distance())
            self.__params['pheromoneMat'][y][x] = self.__params['pheromoneMat'][x][y]


    def updateLocalPheromone(self,ant):
        ant_path = ant.path()
        u = ant_path[-2]
        v = ant_path[-1]        
        self.__params['pheromoneMat'][u][v] = (1 - self.__params['evaporationRate']) * self.__params['pheromoneMat'][u][v] + self.__params['evaporationRate'] * 1
        self.__params['pheromoneMat'][v][u] = self.__params['pheromoneMat'][u][v]

    def chooseNextNode(self,ant):
        q = random.uniform(0,1)
        current_node = ant.path()[-1]
        if q <= self.__params['q0']:
            max_value = 0.0
            max_node = -1
            for node in range(self.__params['noNodes']):
                if ant.visitedNodes()[node] == 0 and self.__params['mat'][current_node][node]:
                    val = self.getPheromoneAndDistanceValueForCurrentEdge(current_node,node)
                    if max_value < val:
                        max_value = val
                        max_node = node
            if(max_node != -1):
                ant.addNodeToPath(max_node)
            return True
        else:
            prob = 0.9
            values = [0] * self.__params['noNodes']
            sum_of_all_values = 0.0
            for node in range(self.__params['noNodes']):
                if ant.visitedNodes()[node] == 0 and self.__params['mat'][current_node][node]:
                    val = self.getPheromoneAndDistanceValueForCurrentEdge(current_node,node)
                    sum_of_all_values += val
                    values[node] = val
            for node in range(self.__params['noNodes']):
                if values[node] != 0:
                    val_per_sum = values[node] / sum_of_all_values
                    if val_per_sum < prob:
                        ant.addNodeToPath(node)
                        return True
            return False

    def getPheromoneAndDistanceValueForCurrentEdge(self,u, v):
        gama = 1 / self.__params['mat'][u][v]
        return math.pow(self.__params['pheromoneMat'][u][v],self.__params['alfa']) * math.pow(gama,self.__params['beta'])

    def bestAnt(self):
        bestAnt = self.__population[0]
        for ant in self.__population:
            if ant.distance() < bestAnt.distance():
                bestAnt = ant
        return bestAnt

    def deleteEdge(self):
        x = random.randint(0, self.__params['noNodes'] - 1)
        y = random.randint(0, self.__params['noNodes'] - 1)

        while x == y and self.__params['mat'][x][y] == 0:
            y = random.randint(0, self.__params['noNodes'] - 1)

        print ("Dynamic graph -> Edge " + str(x) + " - " + str(y) + " has been deleted.")

        self.__params['mat'][x][y] = 0
        self.__params['mat'][y][x] = 0

        self.__params['pheromoneMat'][x][y] = 0
        self.__params['pheromoneMat'][y][x] = 0