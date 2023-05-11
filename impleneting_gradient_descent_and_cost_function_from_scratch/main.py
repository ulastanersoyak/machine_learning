import numpy as np


def cost_function(x,y,weight,bias):
  total_x = x.shape[0]
  cost = 0
  for i in range (total_x):
    guess = weight * x[i] + bias
    cost = cost + (guess -y[i])**2
  return 1/2 *(cost/total_x) 

def gradient_descent(x,y,weight,bias):
    total_x = x.shape[0]
    gd_weight=0
    gd_bias=0
    for i in range(total_x):
      



def main():
    start = 0
    end = 1
    step = 0.02
    X = np.arange(start, end, step)
    start = 1
    end = 2
    step = 0.02
    Y = np.arange(start, end, step)

    weight = 0
    bias = 0

    weight = 0.4
    bias = 0.8
    print(cost_function(X,Y,weight,bias))

if __name__ =="__main__":
   main()