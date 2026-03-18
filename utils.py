from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train = datasets.ImageFolder('dataset/train', transform=transform)
    test = datasets.ImageFolder('dataset/test', transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader