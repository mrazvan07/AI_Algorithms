import warnings
from MyBGDRegression import MyBGDRegression; warnings.simplefilter('ignore')
import csv
import matplotlib.pyplot as plt 
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler

def plot3Ddata(x1Train, x2Train, yTrain, x1Model = None, x2Model = None, yModel = None, x1Test = None, x2Test = None, yTest = None, title = None):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection = '3d')
    if (x1Train):
        plt.scatter(x1Train, x2Train, yTrain, c = 'r', marker = 'o', label = 'train data') 
    if (x1Model):
        plt.scatter(x1Model, x2Model, yModel, c = 'b', marker = '_', label = 'learnt model') 
    if (x1Test):
        plt.scatter(x1Test, x2Test, yTest, c = 'g', marker = '^', label = 'test data')  
    plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()

def loadData(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs

def loadDataMoreInputs(fileName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    return dataNames, data


# extract a particular feature (column)
def extractFeature(allData, names, featureName):
    pos = names.index(featureName)
    return [float(data[pos]) for data in allData]


def standardisation(feature):
    m = sum(feature) / len(feature)
    s = (1 / len(feature) * sum([ (p - m) ** 2 for p in feature])) ** 0.5 
    return [(p - m) / s for p in feature]

# crtDir =  os.getcwd()
# filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
# inputs, outputs = loadData(filePath,'Economy..GDP.per.Capita.', 'Happiness.Score')

# np.random.seed(5)
# indexes = [i for i in range(len(inputs))]
# trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
# testSample = [i for i in indexes  if not i in trainSample]

# trainInputs = standardisation([inputs[i] for i in trainSample])
# trainOutputs = standardisation([outputs[i] for i in trainSample])

# testInputs = standardisation([inputs[i] for i in testSample])
# testOutputs = standardisation([outputs[i] for i in testSample])

# xx = [[el] for el in trainInputs]
# regressor = MyBGDRegression()
# regressor.fit(xx,trainOutputs)
# w0,w1 = regressor.intercept_, regressor.coef_[0]
# print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')

# noOfPoints = 1000
# xref = []
# val = min(trainInputs)
# step = (max(trainInputs) - min(trainInputs)) / noOfPoints
# for i in range(1, noOfPoints):
#     xref.append(val)
#     val += step
# yref = [w0 + w1 * el for el in xref] 

# plt.plot(trainInputs, trainOutputs, 'ro', label = 'training data')  #train data are plotted by red and circle sign
# plt.plot(xref, yref, 'b-', label = 'learnt model')                  #model is plotted by a blue line
# plt.title('train data and the learnt model')
# plt.xlabel('GDP capita')
# plt.ylabel('happiness')
# plt.legend()
# plt.show()

crtDir =  os.getcwd()
filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
names, allData = loadDataMoreInputs(filePath)
feature1 = extractFeature(allData, names, 'Economy..GDP.per.Capita.')
feature2 = extractFeature(allData, names, 'Freedom')
outputs = extractFeature(allData,names,'Happiness.Score')

np.random.seed(5)
indexes = [i for i in range(len(outputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(outputs)), replace = False)
validationSample = [i for i in indexes  if not i in trainSample]

trainInputs1 = standardisation([feature1[i] for i in trainSample])
trainInputs2 = standardisation([feature2[i] for i in trainSample])
trainOutputs = standardisation([outputs[i] for i in trainSample])

validationInputs1 = standardisation([feature1[i] for i in validationSample])
validationInputs2 = standardisation([feature2[i] for i in validationSample])
validationOutputs = standardisation([outputs[i] for i in validationSample])

xx = [[el1,el2] for el1,el2 in zip(trainInputs1,trainInputs2)]

# from sklearn import linear_model
# # model initialisation
# regressor = linear_model.SGDRegressor(alpha = 0.01, max_iter = 100)
# # training the model by using the training inputs and known training outputs
# regressor.fit(xx, trainOutputs)
# # save the model parameters
# w0, w1,w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
# print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1',' + ',w2, ' * x2')

regressor = MyBGDRegression()
regressor.fit(xx,trainOutputs)
w0, w1,w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1',' + ',w2, ' * x2')

noOfPoints = 1000
xref = []
val = min(trainInputs1)
step = (max(trainInputs1) - min(trainInputs1)) / noOfPoints
for i in range(1, noOfPoints):
    xref.append(val)
    val += step

yref = []
val = min(trainInputs2)
step = (max(trainInputs2) - min(trainInputs2)) / noOfPoints
for i in range(1, noOfPoints):
    yref.append(val)
    val += step

zref = [w0 + w1 * el1 + w2 * el2 for el1,el2 in zip(xref, yref)]
computedValidationOutputs = regressor.predict([[x,y] for x,y in zip(validationInputs1,validationInputs2)])
error = 0.0
for t1, t2 in zip(computedValidationOutputs, validationOutputs):
    error += (t1 - t2) ** 2
error = error / len(validationOutputs)
print("prediction error (manual): ", error)


from sklearn.metrics import mean_squared_error
error = mean_squared_error(validationOutputs, computedValidationOutputs)
print("prediction error (tool): ", error)


def plotData3D(validationInputData1,validationInputData2,computedValidationOutputs,realValidationOutputs):
    ax = plt.axes(projection='3d')
    ax.scatter(validationInputData1,validationInputData2,computedValidationOutputs,c=computedValidationOutputs,cmap='Blues',label='Computed Data')
    ax.scatter(validationInputData1,validationInputData2,realValidationOutputs,c=realValidationOutputs,cmap='Greens',label='Real Data')
    plt.xlabel('GDP')
    plt.ylabel('Freedom')
    ax.set_zlabel('Happiness Score')
    plt.legend()
    plt.title('computed validation and real validation data')
    plt.show()

plotData3D(validationInputs1,validationInputs2,computedValidationOutputs,validationOutputs)