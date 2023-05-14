from calendar import c
import imp
import os
from matplotlib import projections
import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def loadDataFromCsvFile(fileName,features,ouputVar):
    raw_data = pnd.read_csv(fileName)
    good_data = raw_data.dropna()

    input_Matrix = []
    for feature in features:
        input_Matrix.append(good_data[feature])
    outputs = good_data[ouputVar]
    return input_Matrix,outputs

features = ['Economy..GDP.per.Capita.','Family']
crtDir =  os.getcwd()
filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
input_Matrix,outputs = loadDataFromCsvFile(filePath,features,'Happiness.Score')


# Split the Data Into Training and Test Subsets
# In this step we will split our dataset into training and testing subsets (in proportion 80/20%).
# Training data set is used for learning the linear model. Testing dataset is used for validating of the model. All data from testing dataset will be new to model and we may check how accurate are model predictions.
np.random.seed(5)
indexes = [i for i in range(len(outputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(outputs)), replace = False)
validationSample = [i for i in indexes  if not i in trainSample]

trainInputs1 = [input_Matrix[0][i] for i in trainSample]
trainInputs2 = [input_Matrix[1][i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]

validationInputs1 = [input_Matrix[0][i] for i in validationSample]
validationInputs2 = [input_Matrix[1][i] for i in validationSample]
validationOutputs = [outputs[i] for i in validationSample]


xx = [[el1,el2] for el1,el2 in zip(trainInputs1,trainInputs2)]
# model initialisation
regressor = linear_model.LinearRegression()
# training the model by using the training inputs and known training outputs
regressor.fit(xx, trainOutputs)
# save the model parameters
w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1',' + ',w2, ' * x2')

# plot the learnt model
# prepare some synthetic data (inputs are random, while the outputs are computed by the learnt model)
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
    plt.ylabel('Family')
    ax.set_zlabel('Happiness Score')
    plt.legend()
    plt.title('computed validation and real validation data')
    plt.show()

plotData3D(validationInputs1,validationInputs2,computedValidationOutputs,validationOutputs)