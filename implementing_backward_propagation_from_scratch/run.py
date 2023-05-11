import numpy as np
from random import random
class multi_layer_perceptron():
    def __init__(self,in_features=3,hidden_layers=[3,5],out_features=2):
        self.in_features=in_features
        self.hidden_layers=hidden_layers
        self.out_features=out_features
        layers = [self.in_features] + self.hidden_layers + [self.out_features]

        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)


        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i],layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives
   
    def forward_propagate(self,inputs):
        activations = inputs
        self.activations[0] = inputs
        
        for i,w in enumerate(self.weights):
            net_inputs = np.dot(activations,w)
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        return activations
    
    def back_propagate(self,error,verbose=False):

        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        # s'(h[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
        # s'(h_[i+1]) = a_[a+1]
        #dE/dW_[i+1] = (y - a_[i+1]) s'(h_[i+1]) W_i s'(h_i) a_(i+1)

        for i in reversed(range(len(self.derivatives))):
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            error = np.dot(delta,self.weights[i].T)
            if verbose:
                print("derivatives for W{}:{}".format(i, self.derivatives[i]))
        return error

    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights +=  derivatives * learning_rate

    def train(self,inputs,targets,epoch_count,learning_rate):
        for i in range(epoch_count):
            sum_error = 0
            for j,input in enumerate(inputs):
                target = targets[j]
                output = self.forward_propagate(inputs=input)
                error = target- output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)
                sum_error+=self._mse(input,output)

            print(f"epoch:{i+1} error:{sum_error/len(inputs)}")
                
    def _mse(self,target,output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self,x):
        return x * (1.0 - x)

    def _sigmoid(self,x):
        return 1/(1 + np.exp(-x))


if __name__ == '__main__':
    mlp = multi_layer_perceptron(2,[5],1)
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    mlp.train(items,targets,50,0.01)
