# CAPRI-CT: Causal Analysis and Predictive Reasoning for Image Quality Optimization in Computed Tomography

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-CC0--1.0-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🔬 Overview

CAPRI-CT is a novel causal-aware deep learning framework designed to optimize image quality in Computed Tomography (CT) imaging while minimizing radiation exposure. The framework integrates image data with acquisition metadata to model underlying causal relationships that influence Signal-to-Noise Ratio (SNR) in CT images.

### Problem Statement
- Optimizing CT image quality while minimizing radiation exposure remains a key clinical challenge
- Traditional methods rely on empirical testing with limited scalability and interpretability
- Conventional deep learning models lack causal understanding of parameter relationships

### Solution
CAPRI-CT addresses these challenges by:
- Modeling causal relationships between imaging parameters and image quality
- Enabling counterfactual inference for "what-if" simulations
- Providing interpretable insights for evidence-based CT protocol optimization

## ✨ Key Features

- **Causal Framework**: Integrates Variational Autoencoders (VAEs) with causal reasoning
- **Ensemble Learning**: Uses multiple model instances for robust predictions
- **Counterfactual Inference**: Enables simulation of parameter changes without physical scans
- **Interpretability**: Provides actionable insights into parameter-quality relationships
- **Uncertainty Quantification**: Delivers predictions with confidence estimates

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/SnehaGeorge22/capri-ct.git
cd capri-ct
```

### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n capri-ct python=3.8
conda activate capri-ct

# Or using venv
python -m venv capri-ct-env
source capri-ct-env/bin/activate  # On Windows: capri-ct-env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Expected Dependencies
```txt
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
opencv-python>=4.5.0
imageio>=2.9.0
tqdm>=4.60.0
```

## 📊 Dataset

### Dataset Description
The dataset comprises CT images of a custom Perspex phantom with corresponding metadata:

- **Images**: 2,759 valid CT images (512×512 pixels, resized to 9×9 for processing)
- **Phantom Design**: 10cm thick Perspex block with 169 cylindrical holes (13×13 grid)
- **Hole Specifications**: Diameters 4-7mm, 5cm deep, 1cm spacing
- **Contrast Agents**: Bismuth nanoparticles (BiNPs) at 50nm and 100nm, Iodine

### Acquisition Parameters
- **Tube Voltages**: 80, 100, 120, 140 kVp
- **Tube Currents**: 215, 430 mAs
- **Rotation Time**: 1.0s
- **Pitch Factors**: 0.35 – 1.5
- **Slice Thickness**: 5mm (fixed)
- **Reconstruction Matrix**: 512×512

### Dataset Structure
```
dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── final_dataset.csv
└── README.md
```

### Metadata Format
The `final_dataset.csv` file contains:
- `filename`: Image filename
- `voltage`: Tube voltage (kVp)
- `current`: Tube current per second (mAs)
- `contrast_agent`: Type of contrast agent used
- `cnr`: Contrast-to-Noise Ratio
- `snr`: Signal-to-Noise Ratio (target variable)

## 🛠 Usage

### Step 1: Data Preprocessing
**What this does:**
- Loads CT images and metadata
- Removes outliers using IQR method
- Resizes images to 9×9 pixels
- Applies data augmentation (rotations, flips)
- Creates stratified train/validation splits (80/20)
- Handles class imbalance through sampling weights

### Step 2: Train CAPRI-CT Model
**Training Process:**
- Initializes ensemble of 5 VAE models with different random seeds
- Uses Adam optimizer with gradient clipping
- Combines reconstruction loss with KL divergence regularization
- Implements early stopping (patience=10)
- Saves best models based on R² score

### Step 3: Evaluate Model Performance
**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Coefficient of Determination (R²)
- Statistical significance tests (Friedman, Wilcoxon)

### Step 4: Perform Causal Inference
**Causal Analysis Features:**
- **Interventions**: Simulate parameter changes (e.g., `do(voltage=120)`)
- **Counterfactuals**: Answer "what-if" questions
- **Uncertainty Quantification**: Provide confidence intervals
- **Visualization**: Generate heatmaps and correlation plots

### Step 5: Generate Predictions with Uncertainty
## 🏗 Model Architecture

### Causal Directed Acyclic Graph (DAG)
The framework models relationships between:
- **Observed Variables**: Voltage (v), Current (t), Contrast Agent (a), CT Image (i)
- **Latent Variable**: Shared representation (z)
- **Target**: Signal-to-Noise Ratio (SNR)

### Deep Structural Equations
1. **Image Generation**: `i = f_i(v, t, a, u_i)`
2. **Latent Representation**: `z = f_z(i, v, t, a, u_z)`
3. **SNR Prediction**: `snr = f_snr(v, t, a, z, u_snr)`

### VAE Architecture Components

#### Encoder
- 3 Convolutional layers (32, 64, 128 filters)
- Batch normalization and ReLU activation
- Dropout for regularization
- Metadata embedding layers (dimensions: v=16, t=8, a=12)
- Feature fusion and latent distribution parameters (μ, σ)

#### Decoder
- Reparameterization trick: `z = μ + σ·ε`
- 2 hidden layers with regression head
- SNR prediction output

#### Ensemble Strategy
- 5 independent model instances
- Different random seeds and data splits
- Aggregated predictions with uncertainty estimates

## 🧪 Experimental Setup

### Data Preprocessing Steps
1. **Quality Control**: Remove corrupt images and invalid metadata
2. **Outlier Detection**: Apply IQR method to filter extreme SNR values
3. **Image Processing**: Resize to 9×9, normalize pixel values
4. **Augmentation**: Random rotations (±10°), horizontal/vertical flips
5. **Stratification**: Balanced train/validation splits by SNR quantiles
6. **Class Balancing**: Duplicate rare cases, apply weighted sampling

### Training Configuration
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with scheduler)
- **Optimizer**: Adam with gradient clipping (max_norm=1.0)
- **Loss Function**: MSE reconstruction + KL divergence
- **Validation**: Smooth L1 loss

### Baseline Comparisons
- CNN Baseline
- ResNet-18
- SqueezeNet
- Statistical significance testing

## 📈 Results

### Model Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **CAPRI-CT (Ensemble)** | **68.03** | **106.49** | **0.799** |
| CNN Baseline | 94.70 | 141.27 | 0.677 |
| ResNet | 89.39 | 129.12 | 0.750 |
| SqueezeNet | 94.25 | 139.20 | 0.687 |

### Key Findings
- **Contrast Agent Impact**: Removing contrast agent reduced R² to 0.005 (most critical factor)
- **Parameter Sensitivity**: Voltage and current removal showed moderate impact
- **Causal Validation**: Interventional scenarios demonstrated realistic SNR changes
- **Uncertainty Quantification**: Higher uncertainty at distribution extremes

### Ablation Study Results
| Model Variant | MAE | RMSE | R² |
|---------------|-----|------|-----|
| Full Model (i,v,t,a) | 68.03 | 106.49 | 0.799 |
| Without Current (i,v,a) | 87.21 | 133.56 | 0.684 |
| Without Voltage (i,t,a) | 91.42 | 139.72 | 0.655 |
| Without Contrast (i,v,t) | 164.26 | 237.14 | 0.005 |
| Image Only (i) | 172.72 | 235.22 | 0.021 |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   ```bash
   git fork https://github.com/SnehaGeorge22/capri-ct.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Include unit tests for new features

4. **Commit Changes**
   ```bash
   git commit -m "Add: detailed description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Development Guidelines
- Use type hints where applicable
- Write comprehensive docstrings
- Add unit tests for new functionality
- Update documentation as needed
- Follow the existing code structure

## 📝 Citation

If you use CAPRI-CT in your research, please cite our paper:

```bibtex
@article{gnanakalavathy2025capri,
  title={CAPRI-CT: Causal Analysis and Predictive Reasoning for Image Quality Optimization in Computed Tomography},
  author={Gnanakalavathy, Sneha George and Razak, Hakim A and Meertens, R and Fieldsend, Jonathan E and Ye, X and Abdelsamea, Mohammed M},
  journal={arXiv preprint arXiv:2507.17420},
  year={2025},
  url={https://arxiv.org/abs/2507.17420}
}
```

**APA Format:**
S. G. Gnanakalavathy, H. A. Razak, R. Meertens, J. E. Fieldsend, X. Ye, and M. M. Abdelsamea, "CAPRI-CT: Causal Analysis and Predictive Reasoning for Image Quality Optimization in Computed Tomography," *arXiv preprint arXiv:2507.17420*, 2025. [Online]. Available: https://arxiv.org/abs/2507.17420

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Sneha George Gnanakalavathy**
- Email: sneha.g220691@gmail.com
- Institution: Department of Computer Science, University of Exeter, UK
- GitHub: [@SnehaGeorge22](https://github.com/SnehaGeorge22)

## 🔗 Links

- **Paper**: [arXiv:2507.17420](https://arxiv.org/abs/2507.17420)
- **Dataset**: [GitHub Repository](https://github.com/SnehaGeorge22/capri-ct)
- **University of Exeter**: [Computer Science Department](https://www.exeter.ac.uk/departments/computerscience/)

---

## 🙏 Acknowledgments

This research was conducted at the University of Exeter. We thank the reviewers from MICCAI International Conference 2025 and BMC Medical Imaging journal for their valuable feedback that strengthened this work.

---

**⭐ If you find this work useful, please consider starring the repository and citing our paper!**
