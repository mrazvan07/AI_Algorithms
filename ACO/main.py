import math
import random
from ACO import ACO
from Ant import Ant
print("start") 
  
def readNet(fileName):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(",")
        for j in range(n):
            mat[i].append(int(elems[j]))
    net['mat'] = mat
    f.close()
    return net

def readTSPFile(filename):
    f = open(filename,'r')
    net = {}
    line = f.readline().strip()

    string = "DIMENSION"
    while(string not in line):
        line = f.readline().strip()
    n = ""
    for char in line:
        if(char.isdigit()):
            n += char
    net['noNodes'] = int(n)

    string = "NODE_COORD_SECTION"
    while(string not in line):
        line = f.readline().strip()

    x_coordinates = []
    y_coordinates = []
    for i in range(net['noNodes']):
        line = f.readline().strip()
        line_split = line.split(" ")
        line_split= [value for value in line_split if value != '']
        x = float(line_split[1])
        y = float(line_split[2])
        x_coordinates.append(x)
        y_coordinates.append(y)

    mat = []
    for node_from in range(net['noNodes']):
        mat.append([])
        for node_to in range(net['noNodes']):
            x1 = x_coordinates[node_from]
            y1 = y_coordinates[node_from]
            x2 = x_coordinates[node_to]
            y2 = y_coordinates[node_to]
            mat[node_from].append(getDistanceByCoordinates(x1,y1,x2,y2))
    net['mat'] = mat
    f.close()
    return net


def getDistanceByCoordinates(x1,y1,x2,y2):
    x_formula = math.pow(x2-x1, 2)
    y_formula = math.pow(y2-y1, 2)
    return math.sqrt(x_formula + y_formula)


#net = readTSPFile("data\dantzig42.tsp")
net = readNet("data\medium.txt")

problParam = {'popSize' : 5, 'dynamic':2}
params = {'noNodes' : net['noNodes'], 'mat' : net['mat']}
params['pheromoneMat'] = []
params['evaporationRate'] = 0.1
params['q0'] = 2
params['alfa'] = 1
params['beta'] = 1

aco = ACO(params,problParam)
aco.initialisation()
aco.initialisationPheromones()

bestOfBestAnt = None

for i in range(10):
    aco.runAco()
    bestAnt = aco.bestAnt()
    #print("Generation " + str(i) + " best ant distance: " + str(bestAnt.distance()) + " | road: " + str(bestAnt.path()))
    if bestOfBestAnt is None or bestAnt.distance() < bestOfBestAnt.distance():
        bestOfBestAnt = bestAnt

    if i % problParam['dynamic'] == 0:
        aco.deleteEdge()
    aco.initialisation()

bestPath = bestOfBestAnt.path().copy()
bestPath.append(bestOfBestAnt.path()[0])
print("Best ant distance: " + str(bestOfBestAnt.distance()) + ", with path: " + str(bestPath))

