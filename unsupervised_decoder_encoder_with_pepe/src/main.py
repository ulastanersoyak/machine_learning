from create_dataset import Pepe_Dataset,create_dataloaders
from encoder import Autoencoder
from test_and_train import train,evaluate,test_with_input
import argparse
from torch import nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('--epochs', type=int, default=0, help='number of times of iteration over dataset')

args = parser.parse_args()

if __name__ == "__main__":
    train_dataset = Pepe_Dataset("data_set\\train")

    test_dataset = Pepe_Dataset("data_set\\test")


    train_dataloader,test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                          test_dataset=test_dataset,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.num_workers)
    
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train(model,train_dataloader,loss_fn,optimizer,device,args.epochs)
    state_dict = torch.load("autoencoder.pth")
    model.load_state_dict(state_dict)
    # evaluate(model,test_dataloader,loss_fn,device)
    test_with_input(model,"huseyin_aktepe.png")