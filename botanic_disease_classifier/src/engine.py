import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
import timeit
def train_one_epoch(model: nn.Module, train_loader: DataLoader, loss_function: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """
    Trains the model for one epoch using the given train loader and returns the average loss and accuracy.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training dataset.
        loss_function (nn.Module): The loss function to compute the training loss.
        optimizer (optim.Optimizer): The optimizer to update the model's parameters.
        device (torch.device): The device (CPU or GPU) to be used for training.

    Returns:
        tuple[float, float]: A tuple containing the average training loss and accuracy.

    """
    model.to(device)
    model.train()
    correct_preds = 0
    train_loss = 0

    for batch_idx,(imgs,targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        preds = model(imgs)

        loss = loss_function(preds,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(targets)
        correct_preds += (preds.argmax(dim=1) == targets).sum().item()

        # print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    total_samples = len(train_loader.dataset)
    train_loss /= total_samples
    train_acc = (correct_preds/total_samples)*100
    return train_loss,train_acc

def test_one_epoch(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, loss_function: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    """
    Perform one epoch of testing on the provided model using the test_loader.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
        loss_function (torch.nn.Module): The loss function to compute the loss.
        device (torch.device): The device (CPU or GPU) to perform the computations on.

    Returns:
        Tuple[float, float]: A tuple containing the average test loss and test accuracy.

    """
    model.to(device)
    model.eval()
    correct_preds = 0
    test_loss = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item() * len(targets)
            correct_preds += (outputs.argmax(dim=1) == targets).sum().item()

        total_samples = len(test_loader.dataset)
        test_loss /= total_samples
        test_acc = (correct_preds / total_samples) * 100
        return test_loss, test_acc
    

def fit_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module,optimizer: optim.Optimizer,epochs: int):
    """
    Train and evaluate the model for the specified number of epochs.

    Args:
        model (nn.Module): The model to be trained and evaluated.
        train_loader (DataLoader): The DataLoader for the training dataset.
        test_loader (DataLoader): The DataLoader for the test dataset.
        loss_function (nn.Module): The loss function to compute the loss.
        optimizer (optim.Optimizer): The optimizer to update the model's parameters.
        epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the training loss, training accuracy, test loss, and test accuracy.
    """

    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"device name -> {torch.cuda.get_device_name(0)}")
        print(f"available VRAM: {torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / (1024**3):.2f} GB")

    train_loss = torch.zeros(epochs)
    train_accuracy = torch.zeros(epochs)
    
    test_loss = torch.zeros(epochs)
    test_accuracy = torch.zeros(epochs)

    min_loss = float('inf')

    print('fitting started')
    for i in range(epochs):
        
        train_start_time = timeit.default_timer()

        train_loss[i],train_accuracy[i] = train_one_epoch(model = model,
                                                     train_loader = train_loader,
                                                     loss_function = loss_function,
                                                     optimizer = optimizer,
                                                     device = device)
        train_end_time = timeit.default_timer() - train_start_time

        test_start_time = timeit.default_timer()

        test_loss[i], test_accuracy[i] = test_one_epoch(model = model,
                                                        test_loader = test_loader,
                                                        loss_function = loss_function,
                                                        device = device)
        test_end_time = timeit.default_timer() - test_start_time

        print(f'epoch{i+1}/{epochs} train loss: {train_loss[i]:.4f} train accuracy: {train_accuracy[i]:.4f}% elapsed time train: {train_end_time:.4f}s test loss: {test_loss[i]:.4f} test accuracy: {test_accuracy[i]:.4f}% elapsed time test: {test_end_time:.4f} epoch runtime:{(train_end_time+test_end_time):.4f}s\n')
        
        if test_loss[i] < min_loss:
            print(f'previous loss was:{min_loss} new loss is: loss{test_loss[i]}\n')
            min_loss = test_loss[i]
            torch.save(model.state_dict(), "best_model.pth")

    return train_loss, train_accuracy, test_loss, test_accuracy
        