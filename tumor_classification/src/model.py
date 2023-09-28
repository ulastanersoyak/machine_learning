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
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (256 -3 + 2*1)/1 +1 = 256
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (256 -3 + 2*1)/1 +1 = 256
            nn.MaxPool2d(kernel_size=2, stride=2))
            # 256/2 = 128

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (128 -3 + 2*1)/1 +1 = 128
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (128 -3 + 2*1)/1 +1 = 128
            nn.MaxPool2d(kernel_size=2, stride=2))
            # 128/2 = 64

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (64 -3 + 2*1)/1 +1 = 64
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (64 -3 + 2*1)/1 +1 = 64
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (64 -3 + 2*1)/1 +1 = 64
            nn.MaxPool2d(kernel_size=2, stride=2))
            # 64/2 = 32

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (32 -3 + 2*1)/1 +1 = 32
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (32 -3 + 2*1)/1 +1 = 32
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # (32 -3 + 2*1)/1 +1 = 32
            nn.MaxPool2d(kernel_size=2, stride=2))
            # 32/2 = 16



        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*16*16, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5)) 
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5))      
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4),
            nn.BatchNorm1d(num_features=4))
        
    def forward(self,x):
        """
        Forward pass through the network
        
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)

        x = flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = nn.LogSoftmax(dim=1)(x)
        return x