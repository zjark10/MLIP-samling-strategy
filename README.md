# MLIP Sampling Strategy

### Comparative Analysis of Structure Sampling Strategies for Robust MLIP Development

This repository contains the implementation and experimental records for evaluating various structure sampling strategies in the development of Machine Learning Interatomic Potentials (MLIPs).

The primary objective is to systematically compare how different configuration space sampling methods affect the generalization performance and stability of MLIP models, with a focus on data efficiency and extrapolation accuracy in unexplored chemical spaces (e.g., sodium-oxide solid electrolytes).

## 📌 Features

- **Systematic Sampling Algorithms**: Implementation of MD-driven, uncertainty-based, and random sampling strategies.
- **MLIP Training Pipeline**: Integrated workflows for training models (e.g., SevenNet) on custom-sampled datasets.
- **Robustness Benchmarking**: Tools for evaluating Force/Energy MAE, RMSE, and stability across diverse test sets.
- **Visualization Tools**: Scripts to visualize the coverage of sampled structures within the configuration space (PCA/t-SNE).

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (Recommended for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/MLIP-sampling-strategy.git
   cd MLIP-sampling-strategy

## Contributing
Contributions are welcome! Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
