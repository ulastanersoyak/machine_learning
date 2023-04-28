import torch as tc
class LinearRegressionModel(tc.nn.Module):
    def __init__(self):
        super().__init__()
        self.weigths = tc.nn.Parameter(tc.randn(1, requires_grad=True,dtype=tc.float ))
        self.bias =  tc.nn.Parameter(tc.randn(1, requires_grad=True, dtype=tc.float ))
        
    def forward(self , x : tc.tensor) -> tc.tensor:
        return self.weigths*x + self.bias