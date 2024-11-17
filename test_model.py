import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.network import MNISTNet
import pytest
import glob
import os

def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found")
    return max(model_files)

def test_model_architecture():
    model = MNISTNet()
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
    except:
        pytest.fail("Model failed to process 28x28 input")
    
    # Test 3: Check output shape
    assert output.shape[1] == 10, f"Model output should have 10 classes, got {output.shape[1]}"

def test_model_accuracy():
    # Force CPU
    device = torch.device('cpu')
    print(f"Testing on: {device}")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load model
    model = MNISTNet().to(device)
    latest_model = get_latest_model()
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"

if __name__ == "__main__":
    test_model_architecture()
    test_model_accuracy() 