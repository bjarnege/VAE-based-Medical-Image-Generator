import medmnist
import torchvision.transforms


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
        return torchvision.datasets.MNIST(root='./data/mnist',
                                          train=(split == "train"),
                                          download=download_dataset,
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]))
    elif dataset_name in medmnist.INFO:
        info = medmnist.INFO[dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
        return DataClass(split=split, transform=transform, download=download_dataset)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
