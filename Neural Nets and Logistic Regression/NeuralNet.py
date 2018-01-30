# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:43:26 2017

@author: lambe
"""

import numpy as np 
import matplotlib.pyplot as plt 

class NeuralNet:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        self.nodes = 5
        self.theta = np.random.randn(input_dim, self.nodes) / np.sqrt(input_dim)
        self.bias = np.zeros((1, self.nodes))
        self.theta2 = np.random.randn(self.nodes, output_dim) / np.sqrt(self.nodes)
        self.bias2 = np.zeros((1, output_dim))
        #print(self.theta)
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        #TODO:
        z = np.dot(X,self.theta) + self.bias
        sig = self.sigmoid(z)
        #print(sig.shape)
        z2 = np.dot(sig, self.theta2) +self.bias2
        exp_z = np.exp(z2)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        X_len = len(X)
        cost_for_sample = np.empty(X_len)
        #print(np.log(softmax_scores))
        for row in range(X_len):
            hot_y = np.zeros(2)
            if y[row] == 0:
                hot_y[0] = 1
            else:
                hot_y[1] = 1
            #print(np.dot(hot_y.T, softmax_scores[row]))
            cost_for_sample[row] = -(np.dot(hot_y.T, np.log(softmax_scores[row])))

        return np.sum(cost_for_sample)/X_len
            

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        #print(len(self.theta))
        #print(len(X))
        #print(len(np.dot(X,self.theta)))
        z = np.dot(X,self.theta) + self.bias
        #exp_z = np.exp(z)
        #softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        sig = self.sigmoid(z)
        z2 = np.dot(sig, self.theta2) +self.bias2
        exp_z = np.exp(z2)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        #print(softmax_scores)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
#        z = np.dot(X,self.theta) + self.bias
#        exp_z = np.exp(z)
#        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
#        predictions = np.argmax(softmax_scores, axis = 1)
#        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO:
        for i in range(5000):
            z = np.dot(X,self.theta) + self.bias
            sig = self.sigmoid(z)
            #print(sig.shape)
            z2 = np.dot(sig, self.theta2) +self.bias2
            exp_z = np.exp(z2)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            #sig2 = self.sigmoid(z2)
            #sig2 = self.sigmoid(z2)
#            ex = np.exp(z2)
#            soft = ex/ np.sum(ex, axis = 1, keepdims = True)            
            
            
            hot_y = np.zeros((1000,2))
            for row in range(len(X)):
                if y[row] == 0:
                    hot_y[row] = [1,0]
                else:
                    hot_y[row] = [0,1]
                    
            #out_error = np.subtract() 
            X_one = np.ones(len(X))
            error = np.subtract(softmax_scores, hot_y)
            grad2w = np.dot(sig.T, error)
            grad2b = np.dot(X_one.T, error)
            
            in_error = np.dot(error, self.theta2.T) * (self.sigmoid_derivative(sig))
            #print(in_error.shape)
            grad1w = np.dot(X.T, in_error)
            grad1b = np.dot(X_one.T, in_error)
            
            learning_rate = 0.005
            self.theta = self.theta - learning_rate * grad1w
            self.theta2 = self.theta2 - learning_rate * grad2w
            self.bias = self.bias - learning_rate * grad1b
            self.bias2 = self.bias2 - learning_rate * grad2b
            #print(self.compute_cost(X,y))
            #print(self.compute_cost(X,y))
        return "Finished Training"

#--------------------------------------------------------------------------
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        #return np.exp(-x)/((1+np.exp(-x))**2)
        return x*(1-x)
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """
    
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################  
