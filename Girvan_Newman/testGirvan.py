from asyncio.windows_events import NULL
import os 
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 
import warnings 
from networkx.algorithms.community.centrality import girvan_newman


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
filePath = os.path.join(crtDir,  'realNetworks\Football', 'football.gml');
communitiesGraph = readGraphFromGmlFile(filePath)


communities = girvan_newman(communitiesGraph)
node_groups = []
for com in next(communities):
    node_groups.append(list(com))

nr_groups = len(node_groups)

# i = 0
# while i!= nr_groups:
#     j = 0
#     size_current_group = len(node_groups[i])
#     while j!=size_current_group:
#         node_groups[i][j]+=1
#         j+=1
#     i+=1
    
available_colors = ['#ff8080', 'blue', '#80bfff', '#29a329', '#ff8c1a', '#e63900',
                      '#ffff33', '#804000', '#003366', '#ff33ff', '#999966', '#80ffff',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
print(node_groups)
 
color_map = []
for node in G.nodes:
    if node in node_groups[0]:
        color_map.append(available_colors[0])
    else:
        color_map.append(available_colors[1])
nx.draw(G, node_color=color_map, with_labels=True)
plt.show()
