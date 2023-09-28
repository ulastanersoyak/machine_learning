import torch
from torch import nn
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
def set_device()-> str:
    """
    Set the device to use for running PyTorch operations, either CPU or GPU.

    Returns:
        str: The name of the device, either "cuda" or "cpu".
    """
    if torch.cuda.is_available():
        print(f"device -> {torch.cuda.device(0)}")
        print(f"device name -> {torch.cuda.get_device_name(0)}")
        print(f"available VRAM: {torch.cuda.get_device_properties(torch.cuda.device(0)).total_memory / (1024**3):.2f} GB")
        return "cuda"
    return "cpu"


def calculate_accuracy(true:int,total:int)->float:
    """Calculates the accuracy of the model

        Args:
            true (int): correct prediction count
            total (int): total prediction count
        Returns:
            accuracy (float): % accuracy of the model
    """
    accuracy = (true/total)*100
    return accuracy

def model_weights_change(model:nn.Module) -> np.ndarray:
    """
    Calculates the change in weights of a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        np.ndarray: An array containing the weight changes.

    """
    weights = np.array([])

    for layer in model.parameters():
        if hasattr(layer, 'weight'):
            weights = np.concatenate((weights,layer.weight.detach().flatten().numpy()))
    return weights





def show_confusion(model, test_dataloader, classes, device):
    y_pred = []
    y_true = []
    model.to(device)
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)

        output = torch.max(torch.exp(output), 1)[1].data.cpu().numpy()
        y_pred.extend(output)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes], columns=[i for i in classes]) 

    accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    title = f"Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%"

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(title)
    plt.show()

