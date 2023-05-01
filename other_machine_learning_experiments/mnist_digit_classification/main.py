import torch
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat_layer = nn.Flatten()
        self.dense_layers = nn.Sequential(nn.Linear(in_features=28*28,out_features=256),
                                          nn.ReLU(),
                                          nn.Linear(in_features=256,out_features=10)
                                          )
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self,input):
        flattened = self.flat_layer(input)
        logits = self.dense_layers(flattened)
        preds = self.softmax_layer(logits)
        return preds


def download_dataset():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    print("mnist downloaded!")
    return train_data,test_data


def create_dataloader(train_data,test_data):
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE)
    return train_dataloader,test_dataloader


def set_device():
    if torch.cuda.is_available():
        print(torch.cuda.device(0))
        print(torch.cuda.get_device_name(0))
        return "cuda"
    return "cpu"

def train_one_epoch(model,train_dataloader,loss_fn,optimizer,device):
    for inputs,targets in train_dataloader:
        
        inputs,targets = inputs.to(device),targets.to(device)
        preds = model(inputs)
        loss = loss_fn(preds,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"loss = {loss.item()}")


def train(model,train_dataloader,loss_fn,optimizer,device,epochs):
    for i in range(epochs):
        start = time.time()
        print(f"epoch = {i+1}")
        train_one_epoch(model,train_dataloader,loss_fn,optimizer,device)
        end = time.time()
        print(f"train epoch {i+1} took {end-start} seconds")
        print("----------------------------------------------------------")
    print("DONE TRAINING!")


def load_model():
    model = Model()
    state_dict = torch.load("model.pth")
    model.load_state_dict(state_dict)
    return model

def predict(model,input,target,class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        index = predictions[0].argmax()
        predicted = class_mapping[index]
        excepted = class_mapping[target]
    return predicted,excepted

if __name__ == "__main__":
    x = input()
    if(x == "1"):
        train_data,test_data=download_dataset() 
        train_dataloader,test_dataloader = create_dataloader(train_data,test_data)
        device = set_device()
        print(f"using device = {device}")
        model = Model().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),LEARNING_RATE)
        train(model,train_dataloader,loss_fn,optimizer,device,EPOCHS)
        torch.save(model.state_dict(),"model.pth")
        print("model saved!")

    else:
        class_mapping = ["0","1","2","3","4","5","6","7","8","9"]
        _,test_data=download_dataset() 
        input,target = test_data[0][0],test_data[0][1]
        model = load_model()
        predicted,excepted = predict(model,input,target,class_mapping)
        print(f"predicted -> {predicted} || expected -> {excepted}")