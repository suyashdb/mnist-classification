import os
import subprocess
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.network import MNISTNet
from train import RotatedMNIST
import pytest
import glob

# Fixtures
@pytest.fixture
def model():
    return MNISTNet()

@pytest.fixture
def sample_batch():
    batch_size = 32
    return torch.randn(batch_size, 1, 28, 28)

@pytest.fixture
def device():
    return torch.device('cpu')

# Helper Functions
def get_latest_model():
    model_files = glob.glob('models/mnist_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found")
    return max(model_files)

def get_python_files():
    """Recursively get all Python files in the project."""
    python_files = []
    for root, _, files in os.walk('.'):
        if 'venv' in root or 'ENV' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

# Original Model Tests
def test_model_architecture(model):
    """Test basic model architecture requirements."""
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

def test_model_accuracy(device):
    """Test model accuracy on test dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
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

# Comprehensive Tests
def test_input_output_shapes(model, sample_batch):
    """Test various input shapes and verify corresponding output shapes."""
    output = model(sample_batch)
    assert output.shape == (32, 10), "Incorrect output shape for batch input"
    
    single_input = torch.randn(1, 1, 28, 28)
    output = model(single_input)
    assert output.shape == (1, 10), "Incorrect output shape for single input"
    
    for batch_size in [1, 4, 16, 64]:
        batch = torch.randn(batch_size, 1, 28, 28)
        output = model(batch)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

def test_forward_pass_values(model, sample_batch):
    """Test the properties of forward pass outputs."""
    output = model(sample_batch)
    
    assert not torch.isnan(output).any(), "Forward pass produced NaN values"
    assert not torch.isinf(output).any(), "Forward pass produced infinite values"
    
    softmax = nn.Softmax(dim=1)
    probs = softmax(output)
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(sample_batch.size(0))), \
        "Softmax probabilities don't sum to 1"

def test_loss_functions(model, sample_batch):
    """Test different loss functions with model outputs."""
    targets = torch.randint(0, 10, (sample_batch.size(0),))
    output = model(sample_batch)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, targets)
    assert not torch.isnan(loss), "CrossEntropyLoss produced NaN"
    assert loss > 0, "Loss should be positive for random predictions"
    
    losses = [nn.CrossEntropyLoss(), nn.NLLLoss()]
    for criterion in losses:
        loss = criterion(output, targets)
        assert torch.is_tensor(loss), f"Loss {criterion.__class__.__name__} failed"
        assert loss.requires_grad, f"Loss {criterion.__class__.__name__} has no gradient"

def test_data_pipeline():
    """Test the data loading and augmentation pipeline."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = RotatedMNIST('./data', train=True, download=True, transform=transform)
    assert len(dataset) == 60000 * 2, "RotatedMNIST should have double the original size"
    
    idx = 100
    img1, label1 = dataset[idx]
    img2, label2 = dataset[idx + 60000]
    
    assert label1 == label2, "Labels should match for original and rotated versions"
    assert img1.shape == img2.shape == (1, 28, 28), "Incorrect image shape"
    assert not torch.equal(img1, img2), "Rotated image should be different from original"

def test_gradients(model, sample_batch):
    """Test gradient computations."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    output = model(sample_batch)
    targets = torch.randint(0, 10, (sample_batch.size(0),))
    
    loss = criterion(output, targets)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Infinite gradient for {name}"
    
    prev_params = {name: param.clone() for name, param in model.named_parameters()}
    optimizer.step()
    
    for name, param in model.named_parameters():
        assert not torch.equal(param, prev_params[name]), f"Parameters {name} not updated"

# Linting Tests
def test_flake8():
    """Test if all Python files pass flake8."""
    python_files = get_python_files()
    cmd = ['flake8', '--max-line-length=100', '--ignore=E402,W503']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Flake8 found issues:\n{result.stdout}\n{result.stderr}")

def test_black():
    """Test if all Python files are formatted according to black."""
    python_files = get_python_files()
    cmd = ['black', '--check', '--line-length=100']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Black formatting issues found:\n{result.stdout}\n{result.stderr}")

def test_isort():
    """Test if imports are properly sorted."""
    python_files = get_python_files()
    cmd = ['isort', '--check-only', '--profile', 'black']
    cmd.extend(python_files)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Import sorting issues found:\n{result.stdout}\n{result.stderr}")

if __name__ == "__main__":
    pytest.main([__file__]) 