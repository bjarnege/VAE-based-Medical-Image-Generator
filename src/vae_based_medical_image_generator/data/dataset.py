import medmnist
import torchvision.transforms
from PIL import Image
from torch.nn.functional import one_hot
import torch
import numpy as np

class MNISTdataset():

    def __init__(self, dataset):
        self.dataset = dataset
        self.imgs = self.dataset.data# [np.array(img).squeeze() for img, label in self.dataset]
        self.labels = [label for img, label in self.dataset]
        self.labels = one_hot(torch.tensor(self.labels)).numpy()

        self.info = {"n_channels": 1,
                    "label": {k: v for k, v in [elem.split(" - ") for elem in dataset.classes]}
                    }
        
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        self.target_transform = None

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of L (L=1 for single-label)
        '''
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img.numpy(), mode="L")

#        if self.as_rgb:
#            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    
def load_dataset(dataset_name: str, split: str, download_dataset: bool = True):
    """
    Loads a MedMNIST dataset with the specified name and split using the medmnist library.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load ("train", "val", or "test").
        download_dataset (bool, optional): Whether to download the dataset if it is not already available locally. Defaults to True.

    Returns:
        medmnist.dataset.MedMNIST: The loaded dataset object.

    Raises:
        KeyError: If the specified dataset name is not found in the medmnist.INFO dictionary.
    """
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                          train=(split == "train"),
                                          download=download_dataset,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor()]
                                          ))
        
        return MNISTdataset(dataset)

    elif dataset_name in medmnist.INFO:
        info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
#                    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

        return DataClass(split=split, transform=transform, download=download_dataset)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
