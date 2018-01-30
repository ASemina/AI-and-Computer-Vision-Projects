# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:42:10 2017

@author: lambe
"""

import numpy as np 
import matplotlib.pyplot as plt 

class LogisticDigit:
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
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
        
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
#        z = np.dot(X,self.theta) + self.bias
#        exp_z = np.exp(z)
#        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
#        #y_mat = np.zeros(softmax_scores.length())
#        #y_mat[y-1] = 1
#        cost = -sum(np.dot(np.log(softmax_scores), y_mat))
#        print(cost/y_mat.length())
#        return cost/y_mat.length()
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
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
#        hot_y = np.zeros((1000,2))
#        for row in range(len(y)):
#            if y[row] == 0:
#                hot_y[row] = [1,0]
#            else:
#                hot_y[row] = [0,1]
#            #print(cost_for_sample)
#        cost_for_sample = np.dot(hot_y.T, np.log(softmax_scores))
        print(len(cost_for_sample), X_len)
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
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    def digit(self,y):
        hot_y = np.zeros((len(y),10))
        #print(hot_y)
        for row in range(len(y)):
            hot_y[row][int(y[row])] = 1
        return hot_y
        
    
    def fit(self,X,y):
        """
        Learns model parameters to fit the data.
        """  
        #TODO:
        for i in range(10000):
            z = np.dot(X,self.theta) + self.bias
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            #grad_w = np.dot(X.T, hot_y - softmaxscores)
            #grad_w = np.empty(len(X))
#            hot_y = np.zeros((1000,2))
#            for row in range(len(X)):
#                if y[row] == 0:
#                    hot_y[row] = [1,0]
#                else:
#                    hot_y[row] = [0,1]
                    
            hot_y = self.digit(y)
            
            grad_w = np.dot(X.T, np.subtract(softmax_scores, hot_y))
            #grad_b = np.dot(np.ones(len(X)).T, hot_y - softmax_scores)
            #grad_b = np.empty(len(X))
            X_one = np.ones(len(X))
            grad_b = np.dot(X_one.T, np.subtract(softmax_scores, hot_y))
            
            self.theta = self.theta - 0.05 * grad_w
            self.bias = self.bias - 0.05 * grad_b
            #print(self.compute_cost(X,y))
        return 0

#--------------------------------------------------------------------------
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