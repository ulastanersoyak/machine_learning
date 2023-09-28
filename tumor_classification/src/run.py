import torch
from data_setup import tumor_dataset,create_dataloaders
from model import tumor_classifier
from engine import train,test,evaluate,test_and_train
import argparse
from torch import nn,optim
import matplotlib.pyplot as plt
from utils import set_device,show_confusion
import os
import torchvision.transforms as transforms
from prettytable import PrettyTable
from torchsummary import summary
PATH = "tumor_classifier.pth"
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 50
MODE = 0
parser = argparse.ArgumentParser(description='hyperparameters')

parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                    help='batch size for dataloaders')

parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                    help='number of workers for dataloaders')

parser.add_argument('--image_size', type=tuple, default=IMAGE_SIZE,
                    help='resolution for images')

parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                    help='learning rate for optimizer')

parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                    help='weight decay for optimizer')

parser.add_argument('--epochs', type=int, default=EPOCHS,
                    help='number of iteration over dataset')

parser.add_argument('--transforms', type=int, default=EPOCHS,
                    help='number of iteration over dataset')

parser.add_argument('--mode', type=int, default=MODE,
                    help='train or eval')
args = parser.parse_args()

if __name__ == "__main__":
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs
    transform_int = args.transforms
    mode = args.mode

     
    train_path= "data\Training"
    test_path= "data\Testing"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        normalize])


    train_dataset= tumor_dataset(root_dir=train_path,image_size=(image_size,image_size),transform=train_transform)
        


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
        
    test_dataset = tumor_dataset(root_dir=test_path,image_size=(image_size,image_size),transform=test_transform)

    train_dataloader,test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                          test_dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          shuffle=True)
    
    if mode == 0:
        model = torch.nn.Module
        if os.path.isfile(PATH):
            print('found trained model.')
            model = tumor_classifier()
            model.load_state_dict(torch.load(PATH))
        else:
            model = tumor_classifier()
            print('created model')

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay)
        device = set_device()
        model.to(device)
        summary(model, input_size=(3, image_size, image_size))

        # train_loss,train_acc,train_time= train(model=model,
        #                                        train_dataloader=train_dataloader,
        #                                        loss_fn=loss_fn,
        #                                        optimizer=optimizer,
        #                                        epochs=epochs,
        #                                        device=device)
        
        # test_loss,test_acc,test_time = test(model=model,
        #                                     test_dataloader=test_dataloader,
        #                                     loss_fn=loss_fn,
        #                                     device=device,
        #                                     epochs=epochs)



        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f'Total Trainable Params: {total_params}')


        train_loss, train_acc, test_loss, test_acc = test_and_train(model=model,
                                                                    train_dataloader=train_dataloader,
                                                                    test_dataloader=test_dataloader,
                                                                    loss_fn=loss_fn,
                                                                    optimizer=optimizer,
                                                                    device=device,
                                                                    epochs=epochs)

        fig, axs = plt.subplots(2, figsize=(10, 8))

        axs[0].plot(range(epochs), train_loss, label='train loss')
        axs[0].plot(range(epochs), test_loss, label='test loss')
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('loss')
        axs[0].set_title('train and test loss')
        axs[0].legend()

        axs[1].plot(range(epochs), train_acc, label='train accuracy')
        axs[1].plot(range(epochs), test_acc, label='test accuracy')
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('accuracy')
        axs[1].set_title('train and test accuracy')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        plt.show()
    if mode == 1:
        evaluate(PATH,test_dataset)

    if mode == 2:
        model = tumor_classifier()
        model.load_state_dict(torch.load(PATH))
        device = set_device()
        show_confusion(model,test_dataloader,test_dataset.headers,device)

    if mode == 3:
        model = tumor_classifier()
        model.load_state_dict(torch.load(PATH))
        model.eval()
        img,label = test_dataset[1]
        pred,convnet1map,convnet2map,convnet3map,convnet4map = model(img.unsqueeze(1))
        convnet1map = convnet1map.detach().squeeze().numpy()
        convnet2map = convnet2map.detach().squeeze().numpy()
        convnet3map = convnet3map.detach().squeeze().numpy()
        convnet4map = convnet4map.detach().squeeze().numpy()

        # Plotting convnet1map
        fig1, axs1 = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            axs1[i // 4, i % 4].imshow(convnet1map[i], cmap='gray')
            axs1[i // 4, i % 4].axis('off')
        fig1.suptitle('Convnet1 Maps')

        # Plotting convnet2map
        fig2, axs2 = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            axs2[i // 4, i % 4].imshow(convnet2map[i], cmap='gray')
            axs2[i // 4, i % 4].axis('off')
        fig2.suptitle('Convnet2 Maps')

        # Plotting convnet3map
        fig3, axs3 = plt.subplots(8, 4, figsize=(12, 18))
        for i in range(32):
            axs3[i // 4, i % 4].imshow(convnet3map[i], cmap='gray')
            axs3[i // 4, i % 4].axis('off')
        fig3.suptitle('Convnet3 Maps')

        # Plotting convnet4map
        fig4, axs4 = plt.subplots(16, 4, figsize=(12, 24))
        for i in range(64):
            axs4[i // 4, i % 4].imshow(convnet4map[i], cmap='gray')
            axs4[i // 4, i % 4].axis('off')
        fig4.suptitle('Convnet4 Maps')

        # Show all the figures
        plt.show()


        plt.tight_layout()
        plt.show()







                
        