import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import MNISTNet
from datetime import datetime
import os
import torchvision.transforms.functional as TF

class RotatedMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(RotatedMNIST, self).__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        # If index >= original length, get rotated version
        original_len = super().__len__()
        is_rotated = index >= original_len
        
        # Get original image and target
        original_index = index if not is_rotated else index - original_len
        img, target = super().__getitem__(original_index)
        
        # Rotate if needed
        if is_rotated:
            img = TF.rotate(img, 90)
            
        return img, target
    
    def __len__(self):
        # Double the dataset size
        return super().__len__() * 2

def train():
    # Force CPU
    device = torch.device('cpu')
    print(f"Training on: {device}")
    
    # Load MNIST dataset with augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use custom dataset class that includes rotated versions
    train_dataset = RotatedMNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"Training with {len(train_dataset)} images (including rotated versions)")
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth', _use_new_zipfile_serialization=False)
    
if __name__ == '__main__':
    train() 