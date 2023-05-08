import torch as tc
from torch import nn
import matplotlib.pyplot as plt
import model


##initialize parameters
###########################
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = tc.arange(start,end,step).unsqueeze(dim=1)
y = weight * X  + bias
###########################



##initialize test/split data
###########################
train_split = int(0.8 * len(X))
X_train , y_train = X[:train_split],y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
###########################

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):

  plt.figure(figsize=(10, 7))

  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  plt.legend(prop={"size": 14})
  plt.show()




def main():
  tc.manual_seed(35)
  model_0 = model.LinearRegressionModel()

  with tc.inference_mode(): 
    y_preds = model_0(X_test)

  print(list(model_0.parameters()))

  print(model_0.state_dict())

  plot_predictions(predictions=y_preds)


  loss_fn = nn.L1Loss()

  train_loss_values = []
  test_loss_values = []
  epoch_count = []

  optimizer = tc.optim.SGD(params=model_0.parameters(),lr=0.01)

  epochs = 1000

  for i in range(epochs):
    model_0.train()

    y_preds = model_0(X_train)

    loss = loss_fn(y_preds,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

    with tc.inference_mode():
      test_pred = model_0(X_test)
      test_loss = loss_fn(test_pred,y_test.type(tc.float))

    train_loss_values.append(loss.detach().numpy())
    test_loss_values.append(test_loss.detach().numpy())
    epoch_count.append(i)
    if i%200 == 0:
      print(f"epoch : {epoch_count[i]}\ntrain loss: {train_loss_values[i]}\ntest loss : {test_loss_values[i]}")

  plot_predictions(predictions=test_pred.detach().numpy())
  plt.plot(epoch_count, train_loss_values, label="Train loss")
  plt.plot(epoch_count, test_loss_values, label="Test loss")
  plt.title("Training and test loss curves")
  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.show()
    

if __name__ ==  "__main__":
  main()
