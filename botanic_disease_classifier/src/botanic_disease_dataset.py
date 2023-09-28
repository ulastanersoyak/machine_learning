import os
from torch.utils.data import Dataset,DataLoader
import imghdr
from PIL import Image
import matplotlib.pyplot as plt
import time
from typing import Optional
import torch
import torchvision.transforms as transforms

class botanic_disease_dataset(Dataset):
    """Dataset class for plant sickness images."""
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, verbose: bool = False):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (transforms.Compose, optional): Optional transform to be applied to the images.
            verbose (bool, optional): Whether to print verbose information. Default is False.
        
        Raises:
            ValueError: If the specified root directory doesn't exist.
        """
        self.cases = []
        self.image_paths = []
        self.transform = transform
        if os.path.exists(root_dir):
            start_time = time.time()
            for plant_type in os.listdir(root_dir):
                self.cases.append(plant_type)
                type_start_time = time.time()
                type_image_count = 0
                for path in os.listdir(os.path.join(root_dir, plant_type)):
                    full_path = os.path.join(root_dir, plant_type, path)
                    if imghdr.what(full_path): #checks whether the file in the path is an image or not
                        self.image_paths.append(full_path)
                        type_image_count += 1
                type_elapsed_time = time.time() - type_start_time
                print(f"loaded {type_image_count} images for {plant_type} in {type_elapsed_time:.2f} seconds.") if verbose else None
            elapsed_time = time.time() - start_time
            print(f"total: loaded {len(self.image_paths)} images for {len(self.cases)} plant types in {elapsed_time:.2f} seconds.") if verbose else None
        else:
            raise ValueError(f"{root_dir} doesn't exist")
    
    def __len__(self) -> int :
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        
        Raises:
            IndexError: If the specified index is out of bounds.
        """
        if index < 0:
            raise IndexError(f'index cant be smaller than 0. youve enterted {index}')
        if index >= len(self.image_paths):
            raise IndexError(f"dataset has {len(self.cases)} cases. {index} is out of bounds")
        path = self.image_paths[index]
        plant_type = os.path.basename(os.path.dirname(path))
        label = self.cases.index(plant_type)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

if __name__ == '__main__':
    a = r'New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
    dataset = botanic_disease_dataset(a,None,True)
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f'{dataset.cases[label]} {label}\n{img.size}')
        plt.axis("off")
        plt.imshow(img)
    plt.show()

    x = DataLoader(dataset,16,True)
        