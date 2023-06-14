from torch import nn
from torch import flatten
class tumor_classifier(nn.Module):
    """
    Neural network model for classifying tumor images into 4 classes.
    
    Args:
    None
    
    Attributes:
    layers (nn.Sequential): Sequential container for neural network layers
    
    Methods:
    forward(x): Forward pass through the network
    
    """
    def __init__(self) -> None:
        super().__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2))
        # (256 -3 + 2*1)/1 +1 = 256/2= 128

        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2))
        # (128 -3 + 2*1)/1 +1 = 128/2= 64

        self.convlayer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2))
        # (64 -3 + 2*1)/1 +1 = 64/2= 32

        self.convlayer4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=2))
        # (32 -5 + 2*1)/1 +1 = 30/2= 15

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=32*15*15, out_features=100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),
            nn.Dropout(0.5))    
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=4),
            nn.BatchNorm1d(num_features=4))
        
    def forward(self,x):
        """
        Forward pass through the network
        
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x