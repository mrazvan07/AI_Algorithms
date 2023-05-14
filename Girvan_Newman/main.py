from asyncio.windows_events import NULL
import os 
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 
import warnings 
from networkx.algorithms.community.centrality import girvan_newman
import scipy as sp

def readGraphFromGmlFile(file):
    if '.txt' in file :
        Graph = nx.read_edgelist(file, comments='%')
        return G
    if '.gml' in file :
        Graph = nx.read_gml(file, label='id');
        return Graph

def greedyGirvanNewman(inputNxGraph, communitiesNumber):
    done = False
    while done != True:
        betweenness_dicitonary = nx.edge_betweenness_centrality(inputNxGraph);
        maxBetweeness = 0
        tupleEdgeToBeRemoved = NULL
        for key,value in betweenness_dicitonary.items():
            if value > maxBetweeness :
                tupleEdgeToBeRemoved = key
                maxBetweeness = value
        inputNxGraph.remove_edge(tupleEdgeToBeRemoved[0], tupleEdgeToBeRemoved[1])
        setsOfCommunititesGenerator = nx.algorithms.connected_components(inputNxGraph)
        setsOfCommunitiesList = list(setsOfCommunititesGenerator)
        if len(setsOfCommunitiesList) == communitiesNumber :
            return [list(elem) for elem in setsOfCommunitiesList]
    return NULL


crtDir =  os.getcwd()
filePath = os.path.join(crtDir,  'realNetworks\Janet', 'JanetBackbone.gml')
communitiesGraph = readGraphFromGmlFile(filePath)

communitiesList = greedyGirvanNewman(communitiesGraph,4)

communities = [0]*nx.number_of_nodes(communitiesGraph)

nr_communities = len(communitiesList)
i = 0
while i != nr_communities-1:
    j = 0
    size_current_community = len(communitiesList[i])
    while j != size_current_community-1:
        communities[communitiesList[i][j]] = i
        j+=1
    i+=1

nr_communities = len(communitiesList)
i = 0
while i != nr_communities:
    j = 0
    size_current_community = len(communitiesList[i])
    while j != size_current_community:
        communitiesList[i][j]+=1
        j+=1
    i+=1

    
for i in range(0,len(communities)):
    communities[i]+=1


print("Numarul de comunitati: "+str(len(communitiesList)))
print(communitiesList)
print(communities)
    

adjacency_matrix = np.matrix(nx.adjacency_matrix(communitiesGraph).todense())
graph = nx.from_numpy_matrix(adjacency_matrix)
pos = nx.spring_layout(graph)  # compute graph layout
plt.figure(figsize=(10, 10))  # image is 8 x 8 inches
nx.draw_networkx_nodes(graph, pos, node_size=70, cmap=plt.cm.RdYlBu, node_color = communities)
nx.draw_networkx_edges(graph, pos, alpha=0.5)
plt.show()

