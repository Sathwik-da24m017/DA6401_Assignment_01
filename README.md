# DA6401_Assignment_01

WANDB Report Link:
https://api.wandb.ai/links/da24m017-indian-institute-of-technology-madras/mrifma0v
Github Repo Link
https://github.com/Sathwik-da24m017/DA6401_Assignment_01/tree/main

# DA6401 - Assignment 1

## Overview
This repository contains the implementation of a feedforward neural network from scratch for DA6401 Assignment 1. The project includes training the model using backpropagation with different optimization techniques and tracking experiments using Weights & Biases (WandB).

## Directory Structure
```
DA6401_Assignment1/
│-- model.py             # Complete implementation
│-- code.ipynb         # All the questions implemented but without required args
│-- Assignment1.ipynb             # Second time implementatin with all the required args
```

## Features
- Implementation of a fully connected neural network with configurable activation functions and loss functions.
- Support for multiple optimization algorithms: 
  - SGD
  - Momentum
  - Nesterov Accelerated Gradient (NAG)
  - RMSProp
  - Adam
  - Nadam
- Training and evaluation scripts with proper logging using WandB.
- Utility functions for data processing and model management.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Sathwik-da24m017/DA6401_Assignment1.git
   cd DA6401_Assignment1
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib wandb
   ```
3. Set up Weights & Biases (optional):
   ```bash
   wandb login
   ```

## Usage
### Training the Model
To train the model, run:
```bash
python train.py --wandb_project <project_name> --wandb_entity <wandb_entity>
```

### Command-Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | Project name in Weights & Biases |
| `-we`, `--wandb_entity` | `myname` | WandB Entity name |
| `-d`, `--dataset` | `fashion_mnist` | Dataset to use (`mnist` or `fashion_mnist`) |
| `-e`, `--epochs` | `10` | Number of training epochs |
| `-b`, `--batch_size` | `32` | Training batch size |
| `-l`, `--loss` | `cross_entropy` | Loss function (`cross_entropy` or `mean_squared_error`) |
| `-o`, `--optimizer` | `adam` | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `-lr`, `--learning_rate` | `0.001` | Learning rate |
| `-m`, `--momentum` | `0.9` | Momentum (for `momentum` and `nag` optimizers) |
| `-beta`, `--beta` | `0.9` | Beta (for `rmsprop`) |
| `-beta1`, `--beta1` | `0.9` | Beta1 (for `adam` and `nadam`) |
| `-beta2`, `--beta2` | `0.999` | Beta2 (for `adam` and `nadam`) |
| `-eps`, `--epsilon` | `1e-8` | Epsilon for optimizers |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay (L2 regularization) |
| `-w_i`, `--weight_init` | `Xavier` | Weight initialization (`random` or `Xavier`) |
| `-nhl`, `--num_layers` | `3` | Number of hidden layers |
| `-sz`, `--hidden_size` | `128` | Number of neurons per hidden layer |
| `-a`, `--activation` | `ReLU` | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |

### Testing the Model
You can test the model using the Jupyter Notebook `Assignment1.ipynb`

### Experiment Tracking
The project utilizes [Weights & Biases (WandB)](https://wandb.ai) for tracking experiments and hyperparameter tuning. To enable experiment tracking, ensure that you are logged into WandB before running `train.py`.

## Model Implementation
The neural network supports the following activation functions:
- Sigmoid
- Tanh
- ReLU
- Identity
- Softmax (for classification output)

### Supported Loss Functions:
- Cross-Entropy Loss (for classification tasks)
- Mean Squared Error (MSE) (for regression tasks)

### Optimizers Implemented:
- Stochastic Gradient Descent (SGD)
- Momentum-Based Gradient Descent
- Nesterov Accelerated Gradient (NAG)
- RMSProp
- Adam
- Nadam

## Results
1. Sample images from the Fashion-MNIST dataset are visualized in `Assignment1.ipynb`.
2. Hyperparameter sweeps using WandB identify optimal configurations.
3. Best performing model is evaluated on the test set, and results are logged in WandB.
4. A confusion matrix is generated for the best model.

