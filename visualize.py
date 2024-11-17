import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

def visualize_mnist_pairs(num_pairs=5):
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Create figure
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 2*num_pairs))
    fig.suptitle('Original vs 90° Rotated MNIST Images', fontsize=14)
    
    # Set column titles
    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Rotated 90°')
    
    # Random indices for visualization
    indices = np.random.randint(0, len(dataset), num_pairs)
    
    for i, idx in enumerate(indices):
        # Get original image and label
        img, label = dataset[idx]
        
        # Create rotated version
        img_rotated = TF.rotate(img, 90)
        
        # Convert to numpy and denormalize for visualization
        img = img.numpy()[0]
        img_rotated = img_rotated.numpy()[0]
        
        # Plot original
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(f'Digit: {label}')
        
        # Plot rotated
        axes[i, 1].imshow(img_rotated, cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('mnist_visualization.png')
    print("Visualization saved as 'mnist_visualization.png'")
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    visualize_mnist_pairs() 