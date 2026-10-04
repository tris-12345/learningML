import math

def sigmoid(x):
    return (1 + 1/(math.exp(-x)))
def activate(input_data, weights):
    weighted_sum = sum(i*w  for i, w in zip(input_data, weights))
    return sigmoid(weighted_sum)

if __name__=="__main__":
    input_data = [0.5, 0.2, 0.1]
    weights=[0.4, 0.3, 0.9]
    outputs = activate(input_data, weights)
    print("Output of the artificial neuron:", outputs)