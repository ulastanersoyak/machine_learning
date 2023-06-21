from torch import nn,flatten,randn,Tensor,cuda
from botanic_disease_dataset import botanic_disease_dataset
from torchsummary import summary

class botanic_disease_recognizer(nn.Module):
    def __init__(self,out_classes:int):
        """
        Initialize the botanic disease recognizer model.

        Args:
            out_classes (int): The number of output classes.
        """
        super().__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace= True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2,stride=2))
        #(128 -3 + 2*1)/1 + 1 = 128/2 = 64

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace= True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2,stride=2))
        #(64 -3 + 2*1)/1 + 1 = 64/2 = 32
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace= True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2,stride=2))
        #(32 -3 + 2*1)/1 + 1 = 32/2 = 16

        # self.convblock4 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace= True),
        #     nn.Dropout(0.2),
        #     nn.MaxPool2d(kernel_size=2,stride=2))
        # #(16 -3 + 2*1)/1 + 1 = 16/2 = 8

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128*16*16,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5))
        
        # self.fc2 = nn.Sequential(
        #     nn.Linear(in_features=1024,out_features=512),
        #     nn.BatchNorm1d(num_features=512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5))
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256,out_features=out_classes),
            nn.BatchNorm1d(num_features=out_classes))
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the botanic disease recognizer model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor with shape (batch_size, out_classes).
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        # x = self.convblock4(x)
        x = flatten(x, start_dim=1)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x

if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'
    dataset = botanic_disease_dataset(r'New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train',None,True)
    model = botanic_disease_recognizer(out_classes=len(dataset.cases))
    model = model.to(device)
    batch_size = 16
    dummy_input = randn(batch_size, 3, 128, 128)
    dummy_input = dummy_input.to(device)
    output = model(dummy_input)
    print(output.shape)
    summary(model, input_size=(3, 128, 128))