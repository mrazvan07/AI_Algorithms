import random
class MyBGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple stochastic GD
    def fit(self, x, y, learningRate = 0.001, noEpochs = 1000):
        #self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients 
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            gradients = [el for el in self.coef_]
            for i in range(len(x)): # for each sample from the training data
                ycomputed = self.eval(x[i])     # estimate the output
                crtError = ycomputed - y[i]     # compute the error for the current sample

                gradients[len(x[0])] += crtError
                for j in range(0, len(x[0])):   # update the gradients
                    gradients[j] += crtError * x[i][j]

            for i in range(len(x[0])+1):
                self.coef_[i] = self.coef_[i] - learningRate * gradients[i] / len(x)
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi 

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed
