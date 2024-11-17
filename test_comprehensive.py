import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.network import MNISTNet
from train import RotatedMNIST
import pytest

@pytest.fixture
def model():
    return MNISTNet()

@pytest.fixture
def sample_batch():
    batch_size = 32
    return torch.randn(batch_size, 1, 28, 28)

# 1. Input/Output Shape Tests
def test_input_output_shapes(model, sample_batch):
    """Test various input shapes and verify corresponding output shapes."""
    # Test batch processing
    output = model(sample_batch)
    assert output.shape == (32, 10), "Incorrect output shape for batch input"
    
    # Test single image
    single_input = torch.randn(1, 1, 28, 28)
    output = model(single_input)
    assert output.shape == (1, 10), "Incorrect output shape for single input"
    
    # Test different batch sizes
    for batch_size in [1, 4, 16, 64]:
        batch = torch.randn(batch_size, 1, 28, 28)
        output = model(batch)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

# 2. Forward Pass Tests
def test_forward_pass_values(model, sample_batch):
    """Test the properties of forward pass outputs."""
    output = model(sample_batch)
    
    # Test output range
    assert not torch.isnan(output).any(), "Forward pass produced NaN values"
    assert not torch.isinf(output).any(), "Forward pass produced infinite values"
    
    # Test if output can be used for softmax
    softmax = nn.Softmax(dim=1)
    probs = softmax(output)
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(sample_batch.size(0))), \
        "Softmax probabilities don't sum to 1"

# 3. Loss Function Tests
def test_loss_functions(model, sample_batch):
    """Test different loss functions with model outputs."""
    # Create dummy targets
    targets = torch.randint(0, 10, (sample_batch.size(0),))
    output = model(sample_batch)
    
    # Test CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, targets)
    assert not torch.isnan(loss), "CrossEntropyLoss produced NaN"
    assert loss > 0, "Loss should be positive for random predictions"
    
    # Test with different loss functions
    losses = [nn.CrossEntropyLoss(), nn.NLLLoss()]
    for criterion in losses:
        loss = criterion(output, targets)
        assert torch.is_tensor(loss), f"Loss {criterion.__class__.__name__} failed"
        assert loss.requires_grad, f"Loss {criterion.__class__.__name__} has no gradient"

# 4. Data Pipeline Tests
def test_data_pipeline():
    """Test the data loading and augmentation pipeline."""
    # Test original MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test RotatedMNIST dataset
    dataset = RotatedMNIST('./data', train=True, download=True, transform=transform)
    
    # Test length
    assert len(dataset) == 60000 * 2, "RotatedMNIST should have double the original size"
    
    # Test original and rotated image pairs
    idx = 100
    img1, label1 = dataset[idx]  # Original
    img2, label2 = dataset[idx + 60000]  # Rotated version
    
    assert label1 == label2, "Labels should match for original and rotated versions"
    assert img1.shape == img2.shape == (1, 28, 28), "Incorrect image shape"
    assert not torch.equal(img1, img2), "Rotated image should be different from original"

# 5. Gradient Tests
def test_gradients(model, sample_batch):
    """Test gradient computations."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Forward pass
    output = model(sample_batch)
    targets = torch.randint(0, 10, (sample_batch.size(0),))
    
    # Test gradient computation
    loss = criterion(output, targets)
    loss.backward()
    
    # Check if gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Infinite gradient for {name}"
    
    # Test gradient update
    prev_params = {name: param.clone() for name, param in model.named_parameters()}
    optimizer.step()
    
    # Verify parameters have been updated
    for name, param in model.named_parameters():
        assert not torch.equal(param, prev_params[name]), f"Parameters {name} not updated"

if __name__ == "__main__":
    pytest.main([__file__]) 