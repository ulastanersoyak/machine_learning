import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


def seperate_train_and_test():
    if(os.path.exists("data_set\data\\train")):
        print("train and test data is already splitted")
        return -1
    
    data_dir = 'data_set'
    dataset_dir = os.path.join(data_dir,"data")
    test_size = 0.2
    file_list = os.listdir(data_dir)

    train_files, test_files = train_test_split(file_list, test_size=test_size)

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        src = os.path.join(dataset_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)

    for file in test_files:
        src = os.path.join(dataset_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy(src, dst)

    print("splitted train and test data")
    return 0

class Pepe_Dataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                filepath = os.path.join(root_dir, filename)
                self.samples.append(filepath)

        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image
    

def create_dataloaders(train_dataset,test_dataset,batch_size,num_workers):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return train_dataloader,test_dataloader