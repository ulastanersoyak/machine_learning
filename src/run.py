import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn,optim,load
import os
from engine import fit_model
from botanic_disease_recognizer import botanic_disease_recognizer
from botanic_disease_dataset import botanic_disease_dataset

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 0.001
    weight_decay = 0.01
    epochs = 50
    path = 'best_model.pth'


    train_path= r"New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
    test_path= r"New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    
    train_dataset = botanic_disease_dataset(root_dir=train_path,transform=transform,verbose=True)
    test_dataset = botanic_disease_dataset(root_dir=test_path,transform=transform,verbose=True)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    model = nn.Module
    if os.path.isfile(path):
        print('found trained model.')
        model = botanic_disease_recognizer(out_classes=len(train_dataset.cases))
        model.load_state_dict(load(path))
    else:
        model = botanic_disease_recognizer(out_classes=len(train_dataset.cases))
        print('created model')
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    train_loss, train_accuracy, test_loss, test_accuracy = fit_model(model=model,train_loader=train_loader,test_loader=test_loader,loss_function=loss_function,optimizer=optimizer,epochs=epochs)