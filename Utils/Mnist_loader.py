import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torchvision.datasets as datasets

def get_mnist_dataloaders(data_dir='../datasets', transform_train=None, transform_test=None, batch_size=64, image_size=192, train_size = 'default'):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform_train, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform_test, download=True)

    if train_size!='default':
        total_train = len(train_dataset)
        temp_test_size = total_train-  int(train_size)
        train_dataset, dataset_temp_test = random_split(train_dataset, [int(train_size), temp_test_size])
        test_dataset = ConcatDataset([dataset_temp_test, test_dataset])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Example usage:
# train_loader, test_loader = get_mnist_dataloaders(data_dir='./dataset', image_size=192)