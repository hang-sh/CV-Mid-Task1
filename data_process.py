from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def get_datasets(data_path, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_val_data, test_data = train_test_split(dataset, test_size=0.2, random_state=random_state)
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=random_state)
    return train_data, val_data, test_data