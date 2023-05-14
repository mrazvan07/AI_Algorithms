
import math
from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import csv
import numpy as np


def readFlowersFile():
    rows = []
    with open("data/flowers.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    realLabels = [el[0] for el in rows] 
    computedLabels = [el[1] for el in rows] 
    return realLabels,computedLabels

def readSportsFile():
    rows = []
    header = []
    with open("data/sports.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)
    input_values = [ [int(x) for x in el[:3]] for el in rows] 
    predicted_values = [ [int(x) for x in el[3:]] for el in rows] 
    input_parameters = header[:3]
    target_parameters = header[3:]
    return input_values,predicted_values,input_parameters,target_parameters
        
# Problem specification:
# input: realOutputs, computedOutputs - arrays of the same length containing arrays of 3 real values representing input values / predicted values
# output: errorL1,errorL2 - real values
def predictionErrorForSportsInput(realOutputs, computedOutputs,no_paramters):
    list_of_L1Errors = []
    list_of_L2Errors = []
    for i in range(len(realOutputs)):
        #MAE (Mean Absolute Error)
        list_of_L1Errors.append(sum(abs(r - c) for r, c in zip(realOutputs[i], computedOutputs[i]))/no_paramters)
        #RMSE (Root Mean Square Error)
        list_of_L2Errors.append(sqrt(sum(pow((r - c),2) for r, c in zip(realOutputs[i], computedOutputs[i]))/no_paramters))
    errorL1 = sum(list_of_L1Errors)/len(realOutputs)
    errorL2 = sum(list_of_L2Errors)/len(realOutputs)
    return errorL1,errorL2
    
def evalMultiClassV1(realLabels, computedLabels, labelNames):
    acc = sum([1 if realLabels[i] == computedLabels[i] else 0 for i in range(0, len(realLabels))]) / len(realLabels)
    TP = {} 
    FP = {}
    TN = {}
    FN = {}
    for label in labelNames:
        TP[label] = sum([1 if (realLabels[i] == label and computedLabels[i] == label) else 0 for i in range(len(realLabels))])
        FP[label] = sum([1 if (realLabels[i] != label and computedLabels[i] == label) else 0  for i in range(len(realLabels))])
        TN[label] = sum([1 if (realLabels[i] != label and computedLabels[i] != label) else 0 for i in range(len(realLabels))])
        FN[label] = sum([1 if (realLabels[i] == label and computedLabels[i] != label) else 0  for i in range(len(realLabels))])

    precision = {}
    recall = {}
    for label in labelNames:
        precision[label] = TP[label]/ (TP[label] + FP[label])
        recall[label] = TP[label] / (TP[label] + FN[label])
        
    return acc, precision, recall 

def evalClassificationV1(realLabels, computedLabels, labelNames):
    accuracy = accuracy_score(realLabels, computedLabels)
    precision = precision_score(realLabels, computedLabels, average=None, labels=labelNames)
    recall = recall_score(realLabels, computedLabels, average=None, labels=labelNames)
    return accuracy, precision, recall

    
input_values, predicted_values,input_parameters,target_parameters = readSportsFile()
realLabels,computedLabels = readFlowersFile()

errorL1,errorL2 = predictionErrorForSportsInput(input_values,predicted_values,3)
print("=================== SPORTS ===================")
print("MAE/L1 = " + str(errorL1))
print("RMSE/L2 = " + str(errorL2))
print("==============================================")
print("\n\n")

accuracy, precision, recall = evalClassificationV1(realLabels, computedLabels, ['Daisy', 'Tulip', 'Rose'])
# a,p,r = evalMultiClassV1(realLabels, computedLabels, ['Daisy', 'Tulip', 'Rose'])
# print(a)
# print(p)
# print(r)
print("=================== FLOWERS ===================")
print("Accuracy = " + str(accuracy))
print("Precision = " + "{ Daisy: " + str(precision[0]) + " ,Tulip: "+ str(precision[1]) + " , Rose: "+ str(precision[2]) + " }")
print("Recall = " + "{ Daisy: " + str(recall[0]) + " ,Tulip: "+ str(recall[1]) + " ,Rose: "+ str(recall[2]) + " }")
print("==============================================")
print("\n")


def evalLogLossV1(realLabels, computedOutputs):
    # suppose that 'smartFella' is the positive class
    realOutputs = [[1, 0] if label == 'smartFella' else [0, 1] for label in realLabels]
    datasetSize = len(realLabels)
    noClasses = len(set(realLabels))
    datasetCE = 0.0
    for i in range(datasetSize):
        sampleCE = - sum([realOutputs[i][j] * math.log(computedOutputs[i][j]) for j in range(noClasses)])
        datasetCE += sampleCE
    meanCE = datasetCE / datasetSize
    return meanCE

realLabels = ['smartFella', 'fartSmella', 'fartSmella', 'fartSmella', 'smartFella', 'fartSmella', 'smartFella','smartFella']
computedOutputs = [ [0.71, 0.29], [0.12, 0.88], [0.75, 0.25], [0.60, 0.40], [0.20, 0.80], [0.30, 0.70],[0.45, 0.55], [0.05, 0.95]]
meanCE = evalLogLossV1(realLabels,computedOutputs)
print("=================== *OPTIONAL* ===================")
print("**Binary** Cross-entropy loss (Logistic Loss/CE loss): " + str(meanCE))
print("==============================================")


def evalSoftmaxCEsample(targetValues, rawOutputs):
    # apply softmax for all raw outputs
    noClasses = len(targetValues[0])
    datasetSize = len(targetValues)
    dataSetCE = 0.0
    for i in range(datasetSize):
        expValues =[math.exp(val) for val in rawOutputs[i]]
        sumExpVal = sum(expValues)
        mapOutputs = [val / sumExpVal for val in expValues]
        print(mapOutputs, ' sum: ', sum(mapOutputs))
        sampleCE = - sum([targetValues[i][j] * math.log(mapOutputs[j]) for j in range(noClasses)])
        dataSetCE += sampleCE
    meanCE = dataSetCE / datasetSize
    return meanCE

targetValues = [[0,1,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,1,0,0,0]]
rawOutputs = [[-0.5, 1.2, 0.1, 2.4, 0.3],[0.5, 2.2, 0.1, 0.4, 0.3],[1.0, 2.2, 3.1, 2.4, 0.3],[-0.5, 1.2, 0.1, 2.4, 0.3],[0.5, 2.2, -0.1, 0.4, -0.3]]

print("=================== *OPTIONAL* ===================")
print("**Multi-Class** Cross-entropy loss (Logistic Loss/CE loss): \n")
evalSoftmaxCEsample(targetValues,rawOutputs)
print("==============================================")

def evalSigmoidCEsample(targetValues, rawOutputs):
    # apply softmax for all raw outputs
    noClasses = len(targetValues[0])
    datasetSize = len(targetValues)
    dataSetCE = 0.0
    for i in range(datasetSize):
        mapOutputs = [1 / (1 + math.exp(-val)) for val in rawOutputs[i]]
        print(mapOutputs, ' sum: ', sum(mapOutputs))
        sampleCE = - sum([targetValues[i][j] * math.log(mapOutputs[j]) for j in range(noClasses)])
        dataSetCE += sampleCE
    meanCE = dataSetCE / datasetSize
    return meanCE

targetValues = [[0,1,0,0,1],[1,0,1,0,1],[0,0,1,0,0],[1,0,0,0,1],[0,1,0,1,0]]
rawOutputs = [[-0.5, 1.2, 0.1, 2.4, 0.3],[0.5, 2.2, 0.1, 0.4, 0.3],[1.0, 2.2, 3.1, 2.4, 0.3],[-0.5, 1.2, 0.1, 2.4, 0.3],[0.5, 2.2, -0.1, 0.4, -0.3]]
print("=================== *OPTIONAL* ===================")
print("**Multi-Label** Cross-entropy loss (Logistic Loss/CE loss): \n")
evalSigmoidCEsample(targetValues,rawOutputs)
print("==============================================")
