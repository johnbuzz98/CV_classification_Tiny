import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image

class TinyImageNet200(Dataset):
    def __init__(self, train: bool = True, transform: Optional[Callable] = None, size=(32, 32)):
        super().__init__()
        self.train = train
        self.transform = transform
        self.size = size
        self.data = open("./datasets/tiny_dataset_200_" + ("train" if self.train else "val") + ".txt").read().splitlines()
        self.targets = [x[:9] for x in self.data]
        self.class_to_idx = open("./data/wnids.txt").read().splitlines()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.train:
            img_path = os.path.join("./data/train/", self.data[index])
        else:
            img_path = os.path.join("./data/val/", self.data[index][10:])

        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.size)
        target = self.targets[index]

        if img.size != self.size:
            img.resize(self.size)

        

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        #img = transforms.ToTensor()(img).clone().detach().requires_grad_(True)
        target = torch.tensor(self.class_to_idx.index(target), dtype=torch.long)

        return img, target
        
    
    def __len__(self):
        return len(self.data)


def create_dataset(input_size: Tuple[int, int], aug_name: str = 'default'):
    trainset = TinyImageNet200(
        train     = True,
        transform=__import__('datasets').__dict__[f'{aug_name}_augmentation'](),
        size = input_size
    )
    testset = TinyImageNet200(
        train     = False,
        transform=__import__('datasets').__dict__['test_augmentation'](),
        size = input_size
    )

    return trainset, testset

def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2
    )