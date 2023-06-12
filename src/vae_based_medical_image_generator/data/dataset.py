import medmnist
import torchvision.transforms 

def load_dataset(dataset_name: str, split: str, download_dataset: bool = True):
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    return DataClass(split=split, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.5], std=[.5])
    ]), download=download_dataset)
