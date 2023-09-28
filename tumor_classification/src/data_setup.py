import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
class tumor_dataset(Dataset):
    """
    A dataset class for tumor images.
    """
    def __init__(self,root_dir:str,
                 image_size:Tuple[int,int],
                 transform: Optional[Callable]) -> None:
        """
        Initializes a dataset from the given root directory containing images.

        Args:
            root_dir (str): The relative path of the folder containing the images.
            image_size (Tuple[int, int]): The size that the images will be set to.
            transform (Optional[Callable]): Transforms to apply on the dataset
        Raises:
            ValueError: If the given root directory does not exist.

        Returns:
        None
        """
        if(os.path.exists(root_dir)):
            self.headers = []
            self.cases = []
            self.image_size = image_size
            self.transform = transform
            for tumor_type_dir in os.listdir(root_dir):
                self.headers.append(tumor_type_dir)
                for path in os.listdir(os.path.join(root_dir, tumor_type_dir)):
                    if(path[-4:]==".jpg" or path[-4:]==".png"): #checks if path of the file ends with .png or .jpg by looking at the last 4 elements in the path name.
                        self.cases.append(os.path.join(root_dir, tumor_type_dir, path))
        else:
            raise ValueError(f"path {root_dir} doesn't exist")

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Args:
            None

        Returns:
            int: The number of cases in the dataset.
        """
        return len(self.cases)
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor,int]:
        """
        Returns the image and label at the given index of the dataset.
    
        Args:
        index (int): Index of the sample to retrieve.

        Returns:
        Tuple[Image,int]: A tuple containing the image and corresponding label.

        Raises:
        IndexError:If given index is bigger than case count
        """
        if(index>len(self.cases)):
            raise IndexError(f"dataset has {len(self.cases)} cases. {index} is out of bounds")
        path = self.cases[index]
        header = os.path.basename(os.path.dirname(path))
        img = Image.open(path).resize(self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        # img = img.to(torch.float16)
        label= self.headers.index(header)
        return img,label
    
    def visualize_case(self,index:int)->None:
        """
        Visualizes the image at given index

        Args:
        index (int): Index of the sample to visualize.

        Returns:
        None

        Raises:
        IndexError:If given index is bigger than case count
        """
        if(index>len(self.cases)):
            raise IndexError(f"dataset has {len(self.cases)} cases. {index} is out of bounds")
        path = self.cases[index]
        header = os.path.basename(os.path.dirname(path))
        label= self.headers.index(header)
        img = Image.open(path).resize(self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img)
        img = img.transpose((1,2,0))
        plt.title(f"header:{header}\nlabel:{label}\nsize:{self.image_size}")
        plt.imshow(img)
        plt.show()

def create_dataloaders(train_dataset:Dataset,test_dataset:Dataset,batch_size:int,num_workers:int,shuffle:bool) -> Tuple[DataLoader,DataLoader]:
    """
    Creates PyTorch DataLoader objects from given training and test datasets with the specified batch size,
    number of workers, and shuffle option.

    Args:
        train_dataset (Dataset): The training dataset.
        test_dataset (Dataset): The test dataset.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of worker processes used to load data in parallel.
        shuffle (bool): If True, the DataLoader will shuffle the samples before each epoch.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple of two DataLoader objects, one for training and one for testing.

    Raises:
        Raises an exception if anything goes wrong. Check the terminal for more detailed info about the exception.
    """
    try:
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers)
        test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        print("created dataloaders")
        return train_dataloader,test_dataloader
    except Exception as e:
        print(f"An error occurred while creating the dataloaders: {e}")
        return None