import random

class Ant:
    def __init__(self, params = None):
        self.__params = params
        self.__startNode = random.randint(0,params['noNodes']-1)
        self.__path = [self.__startNode]
        self.__visitedNodes = [0] * self.__params['noNodes']
        self.__visitedNodes[self.__startNode] = 1
    
    
    def path(self):
        return self.__path
    
    
    def startNode(self):
        return self.__startNode
    
    
    def visitedNodes(self):
        return self.__visitedNodes

    
    def startNode(self, node = 0):
        self.startNode = node
    

    def addNodeToPath(self, node):
        self.__path.append(node)
        self.__visitedNodes[node] = 1
    
    def distance(self):
        dist = 0
        if len(self.__path) != self.__params['noNodes']:
            return 99999
        for i in range(len(self.__path)-1):
            if self.__params['mat'][self.__path[i]][self.__path[i+1]] == 0 or self.__params['mat'][self.__path[len(self.__path)-1]][self.__path[0]] == 0:
                return 99999
            dist += self.__params['mat'][self.__path[i]][self.__path[i+1]]
        dist += self.__params['mat'][self.__path[len(self.__path)-1]][self.__path[0]]
        return dist

    def reinitialize(self):
        self.__path = [self.__startNode]
        self.__visitedNodes = [0] * self.__params['noNodes']
        self.__visitedNodes[self.__startNode] = 1





    