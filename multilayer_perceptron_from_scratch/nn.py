import numpy as np

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
    
    def forward_propagate(self,inputs):
        activations = inputs
        
        for w in self.weights:
            net_inputs = np.dot(activations,w)
            activations = self._sigmoid(net_inputs)
        
        return activations
    
    def _sigmoid(self,x):
        return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    mlp = multi_layer_perceptron()

    inputs = np.random.rand(mlp.in_features)

    outputs = mlp.forward_propagate(inputs)
    print(f"input ->{inputs}")
    print(f"output ->{outputs}")