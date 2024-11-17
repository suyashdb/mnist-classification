# MNIST Classification with CI/CD Pipeline

This project implements a simple Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model architecture is designed to be lightweight (<25,000 parameters) while maintaining >80% accuracy.

## Project Structure 
.
├── model/
│ └── network.py # CNN architecture definition
├── .github/
│ └── workflows/ # GitHub Actions CI/CD configuration
├── train.py # Training script
├── test_model.py # Testing and validation script
├── requirements.txt # Python dependencies
└── README.md # This file


## Model Architecture
- Input Layer: Takes 28x28 grayscale images
- 2 Convolutional layers with max pooling
- 2 Fully connected layers
- Output: 10 classes (digits 0-9)
- Total parameters: ~15,418

## Features
- Automated model training
- Automated testing pipeline
- Model validation checks:
  - Parameter count < 25,000
  - Input shape compatibility (28x28)
  - Output shape verification (10 classes)
  - Accuracy threshold > 80%
- Model versioning with timestamps
- Artifact archiving in GitHub Actions

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest
- numpy (< 2.0.0)

## Local Setup

1. Clone the repository
```bash
git clone https://github.com/suyashdb/mnist-classification.git
```

2. Create and activate a virtual environment:
```bash
python -m venv ENV
source ENV/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the training script:
```bash
python train.py
```
This will:
- Download MNIST dataset (if not present)
- Train the model for 1 epoch
- Save the model with timestamp in `models/` directory

2. Run tests:
```bash
python test_model.py
```
This will verify:
- Model architecture constraints
- Model performance on test set

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Archives the trained model as an artifact

The pipeline triggers on every push to the repository.

## Model Artifacts

Trained models are saved with timestamps in the format:
```
models/mnist_model_YYYYMMDD_HHMMSS.pth
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch: https://pytorch.org/