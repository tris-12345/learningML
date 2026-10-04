import numpy as np
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

        
    def forward_propogate(self,inputs):
        #1. calculate net inputs
        #2. calculate actiation
        activations=inputs
        for weight in self.weights:
            net_inputs = np.dot(activations,weight)
            activations = self.sigmoid(net_inputs)
 
        return activations
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

if __name__ == "__main__":
    mlp = MLP()
    inputs = np.array([1,2,3])
    print("Inputs:", inputs)
    outputs = mlp.forward_propogate(inputs)
    print("Outputs: ",outputs)

