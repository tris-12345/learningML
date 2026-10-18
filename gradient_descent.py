#1. activations and derivatives
#2. implement backpropagation algorithm
#3. implement gradient descent
#4. implement training
#5. train neural network with dummy datset
#6. predictions

import numpy as np
from random import random
class MLP:
    def __init__(self, num_inputs=3,num_hidden=[3,5],num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden + [num_outputs]
        self.weights = []
        for i in range(len(layers)-1):
            weight_matrix = np.random.rand(layers[i],layers[i+1])
            self.weights.append(weight_matrix)

        self.activations = []
        for i in range(len(layers)):
            activation_matrix = np.zeros([i])
            #i is the number of neurons in the layer
            self.activations.append(activation_matrix)
        
        self.derivatives = []
        for i in range(len(layers)-1):
            derivative_matrix = np.zeros((layers[i],layers[i+1]))
            #i is the number of neurons in the layer
            self.derivatives.append(derivative_matrix)
        
        
    def forward_propogate(self,inputs):
        #1. calculate net inputs
        #2. calculate activation
        activations=inputs
        self.activations[0]= inputs
        #input of the first layer
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations,w)
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        #a_3 = sigmoid(w_2*a_2)
        return activations
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def back_propogate(self, error,verbose =False):
        for i in reversed(range(len(self.derivatives))):
            activations =self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            #ndarray[0.1,0.2,0.3] to a 2d array-vertical vector [[0.1],[0.2],[0.3]] and then transpose it
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            current_activations =self.activations[i]
            #rearrange the current activations ndarray[0.1,0.2,0.3] to a 2d array-vertical vector [[0.1],[0.2],[0.3]]
            self.derivatives[i]=np.dot(current_activations.reshape(current_activations.shape[0],-1), delta_reshaped)  
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}:{}".format(i,self.derivatives[i]))

        return error

 
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            #print("original weights for W{}:{}".format(i,self.weights[i]))
            weights = self.weights[i]
            derivatives= self.derivatives[i]
            weights +=derivatives*learning_rate
            #print("updated weights for W{}:{}".format(i,weights))
    
    def train(self,inputs, targets,epochs, learning_rate):
        #inputs->network ->outputs : epochs- number of times we want to feed it to the neural network
        
        for i in range(epochs):
            sum_error=0
            for (input, target) in zip(inputs,targets):
                output =self.forward_propogate(input)
                error= target-output
                self.back_propogate(error)
                self.gradient_descent(learning_rate)

                #report the error for each epoch. mse- mean sq e
                sum_error+=self._mse(target,output)

            print("Error: {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target,output):
        return np.average((target-output)**2)






if __name__ == "__main__":
    mlp = MLP(2,[5],1)
    #compute sum operation

    #create a dummy dataset of 1000 items with 2 inputs and 1 output
    items =np.array([[random()/2 for i in range(2)] for i in range(1000)])
    #target is the sum of 2 inputs
    targets = np.array([[item[0]+item[1]] for item in items])

    mlp.train(items,targets,50,0.1)

    # #forwad propogation
    # output = mlp.forward_propogate(inputs)
    # print("Output: ",output)
    # #error calculation
    # error = target-output
    # print("Error: ",error)
    # #back propogation
    # mlp.back_propogate(error,verbose = True)
    # #apply gradient descent
    # mlp.gradient_descent(learning_rate=0.1)



