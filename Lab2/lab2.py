## ----------------------- DataSETS ---------------------------- ##
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()

iris = datasets.load_iris()
x_train = iris.data
y = iris.target
y_train = np.zeros([len(y), 1])
for i in range(len(y)):
    y_train[i][0] = y[i]

x_train = normalize(x_train)

# X = (hours sleeping, hours studying), y = Score on test

#X = np.array(([0,0], [0,1], [1,0], [1,1]), dtype=float)
#y = np.array(([0], [1], [1], [1]), dtype=float)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))

print(x_train)
print(y_train)
# Normalize
#X = X
#y = y #Max test score is 100

## -------------------- Activation Functions ----------------- ##

class Sigmoid(object):
    def __init__(self):
        return
    
    def function(self, x):
        return 1/(1+np.exp(-x))
    
    def derivate(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)        
        
class ReLU(object):

    def __init__(self):
        return

    def relu(self, x):
        if x < 0:
            return 0
        if x >= 0:
            return x

    def drelu(self, x):
        if x < 0:
            return 0
        if x >= 0:
            return 1
    
    def function(self, x):
        x = np.vectorize(self.relu)(x)
        return x
    
    def derivate(self, x):
        x = np.vectorize(self.drelu)(x)
        return x

## ----------------------- Loss Functions -------------------- ##
    

class Neural_NetworkMSE(object):
    def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize, activationFunction):        
        #Define Hyperparameters
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.activationFunction = activationFunction
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activationFunction.function(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activationFunction.function(self.z3) 
        return yHat
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = (1/(self.inputLayerSize))*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.activationFunction.derivate(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.activationFunction.derivate(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class Neural_NetworkAE(object):
    def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize, activationFunction):        
        #Define Hyperparameters
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.activationFunction = activationFunction
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.activationFunction.function(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activationFunction.function(self.z3) 
        return yHat
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = (1/(self.inputLayerSize))*sum(abs(y-self.yHat))
        return J

    def dabs(self, y, yHat):
        if y > yHat:
            return 0
        if y < yHat:
            return 1
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        v = np.vectorize(self.dabs)(y,self.yHat)
        
        delta3 = np.multiply(-(v), self.activationFunction.derivate(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.activationFunction.derivate(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad
        
## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res



S = Sigmoid()
#R = ReLU()
NN = Neural_NetworkMSE(4, 1, 8, S)
#NN2 = Neural_NetworkAE(4, 1, 3, S)
T = trainer(NN)
T.train(x_train,y_train)
#T2 = trainer(NN2)
#T2.train(X,y)
