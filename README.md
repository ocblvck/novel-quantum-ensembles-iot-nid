# Quantum Machine Learning for IoT Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-blueviolet.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive evaluation of **Quantum Machine Learning (QML)** for IoT network intrusion detection, comparing **25 distinct model configurations** across quantum kernel methods, variational quantum algorithms, and novel quantum ensemble techniques.

## 🎯 Project Overview

This repository contains the experimental code and dataset for our research paper:

> **"Comprehensive Evaluation of Quantum Machine Learning for IoT Network Intrusion Detection: Novel Ensemble Methods and GPU-Accelerated Analysis"**
> 
> *Submitted to IEEE DCAS 2026*

### Research Objectives

1. **Systematic Comparison**: Evaluate 25 ML models (16 quantum + 9 classical) across multiple configurations
2. **Novel Ensemble Methods**: Introduce **Quantum Voting Ensemble (QVE)** and **Quantum Weighted Ensemble (QWE)** - techniques unexplored in prior quantum security literature
3. **Scalability Analysis**: Investigate qubit scaling (10 → 16 qubits) and sample size impact (5K → 10K samples)
4. **GPU Acceleration**: Leverage multi-GPU parallel processing for practical quantum simulation
5. **NISQ Viability**: Demonstrate practical quantum advantage boundaries for IoT security

## 📊 Key Results

| Model | Accuracy | F1-Score | MCC | Training Time |
|-------|----------|----------|-----|---------------|
| **Quantum Voting Ensemble (QVE)** | **99.53%** | **0.9953** | **0.9940** | 18.72s |
| QSVC (Z-Feature Map) | 99.43% | 0.9943 | 0.9928 | 16.58s |
| Quantum Random Forest | 93.37% | 0.9337 | 0.9118 | 19.99s |
| Random Forest (Classical) | 99.70% | 0.9970 | 0.9961 | 0.39s |

**Key Findings:**
- ✅ QVE achieves **99.53% accuracy** - highest among all quantum models
- ✅ Z-feature map outperforms ZZ/Pauli by **3.73%** with minimal circuit depth
- ✅ Precomputed kernel achieves **600,000× speedup** over per-sample computation
- ✅ All quantum kernels achieve **perfect specificity (1.00)** - zero false positives

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENTAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌────────────────────────────────────────┐                 │
│  │ IoTID20  │───▶│         PREPROCESSING PIPELINE          │                │
│  │ Dataset  │    │  SelectKBest → MinMaxScaler → PCA       │                │
│  │ 625,783  │    │  (MI, 2n)      [0, π]       (n comp)    │                │
│  │ samples  │    └────────────────────────────────────────┘                 │
│  └──────────┘                        │                                       │
│                                      ▼                                       │
│              ┌───────────────────────┴───────────────────────┐              │
│              │                                               │              │
│              ▼                                               ▼              │
│  ┌───────────────────────┐                    ┌──────────────────────────┐  │
│  │  QUANTUM MODELS (16)  │                    │  CLASSICAL MODELS (9)    │  │
│  ├───────────────────────┤                    ├──────────────────────────┤  │
│  │ • QSVC (8 variants)   │                    │ • SVM (Linear/RBF/Poly)  │  │
│  │   - Z/ZZ/Pauli maps   │                    │ • Random Forest          │  │
│  │   - Standard/Precomp  │                    │ • Gradient Boosting      │  │
│  │   - PegasosQSVC       │                    │ • Logistic Regression    │  │
│  │ • VQC (3 optimizers)  │                    │ • KNN, Decision Tree     │  │
│  │ • QNN (2 variants)    │                    │ • Gaussian Naive Bayes   │  │
│  │ • Ensembles (3)       │                    └──────────────────────────┘  │
│  │   - QVE, QWE, QRF     │                                                  │
│  └───────────────────────┘                                                  │
│              │                                               │              │
│              └───────────────────────┬───────────────────────┘              │
│                                      ▼                                       │
│                    ┌─────────────────────────────────┐                      │
│                    │       EVALUATION METRICS        │                      │
│                    │  Accuracy, Precision, Recall,   │                      │
│                    │  F1, Specificity, MCC, Time     │                      │
│                    └─────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (RTX 3000/4000/A-series recommended)
- **48GB+ GPU VRAM** for 16-qubit simulations (can run smaller configs with less)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ocblvck/quantum-ml-iot-nid.git
cd quantum-ml-iot-nid
```

2. **Create a virtual environment:**
```bash
conda create -n qml-iot python=3.10
conda activate qml-iot
```

3. **Install dependencies:**
```bash
pip install numpy pandas scikit-learn
pip install qiskit qiskit-aer-gpu qiskit-machine-learning qiskit-algorithms
pip install ray[default]  # Optional: for distributed computing
pip install pynvml psutil  # Optional: for GPU monitoring
```

4. **Download the dataset:**
   
   The IoTID20 dataset is available at: [IoTID20 on Kaggle](https://www.kaggle.com/datasets/subhajournal/iotid20-iot-botnet-dataset)
   
   Place the CSV file in the repository root as `IoT_Original_Distribution.csv`

### Usage

#### Basic Run (10 qubits, 5000 samples)
```bash
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group all
```

#### Run Specific Model Groups
```bash
# Quantum SVC models only
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group qsvc

# Variational Quantum Classifiers
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group vqc

# Quantum ensembles (QVE, QWE, QRF)
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group ensemble

# Classical baselines
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group classical
```

#### Advanced Configuration
```bash
# 16 qubits with specific GPU settings
python iot_multigpu.py --num_qubits 16 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv --model_group all \
    --max_gpus 4 --transpile-opt-level 2 --gpus-per-model 2

# Run specific models only
python iot_multigpu.py --num_qubits 10 --sample_size 5000 \
    --dataset IoT_Original_Distribution.csv \
    --only-models "QSVC_Z,QSVC_ZZ,Quantum_Voting_Ensemble"
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_qubits` | Number of qubits (equals feature dimensions) | **Required** |
| `--sample_size` | Number of samples to use from dataset | **Required** |
| `--dataset` | Path to the CSV dataset | **Required** |
| `--model_group` | Model category: `all`, `quantum`, `qsvc`, `vqc`, `qnn`, `ensemble`, `classical` | `qsvc` |
| `--max_gpus` | Maximum number of GPUs to use | Auto-detect |
| `--transpile-opt-level` | Circuit optimization level (0-3) | `2` |
| `--gpus-per-model` | GPUs dedicated per model | `1` |
| `--only-models` | Comma-separated list of specific models to run | None |
| `--distributed-ray` | Enable Ray-based distributed execution | `False` |
| `--ray-address` | Ray cluster address for multi-node | None |

## 📁 Project Structure

```
quantum-ml-iot-nid/
├── iot_multigpu.py              # Main experiment script
├── IoT_Original_Distribution.csv # IoTID20 dataset (download separately)
├── README.md                    # This file
├── LICENSE                      # MIT License
├── results/                     # Output CSVs with metrics
│   ├── quantum_ml_results_10q_5000s.csv
│   ├── quantum_ml_results_10q_10000s.csv
│   └── quantum_ml_results_16q_5000s.csv
├── checkpoints/                 # Model checkpoints
├── gpu_logs/                    # GPU utilization logs
└── kernel_cache/                # Precomputed kernel matrices
```

## 🔬 Models Evaluated

### Quantum Models (16)

| Category | Models | Description |
|----------|--------|-------------|
| **QSVC** | Z, ZZ, Pauli | Quantum kernel SVMs with different feature maps |
| **QSVC Variants** | Precomputed, Callable, Standard | Different kernel computation strategies |
| **PegasosQSVC** | Z, ZZ | Stochastic gradient descent quantum SVM |
| **VQC** | COBYLA, SPSA, ADAM | Variational quantum classifier with different optimizers |
| **QNN** | EstimatorQNN, SamplerQNN | Quantum neural networks |
| **Ensembles** | QVE, QWE, QRF | Novel quantum ensemble methods |

### Classical Baselines (9)

SVM (Linear, RBF, Polynomial), Random Forest, Gradient Boosting, Logistic Regression, KNN, Decision Tree, Gaussian Naive Bayes

## 📈 Experimental Configurations

| Config | Qubits | Samples | Purpose |
|--------|--------|---------|---------|
| 10q/5K | 10 | 5,000 | Baseline performance |
| 10q/10K | 10 | 10,000 | Sample size scaling |
| 16q/5K | 16 | 5,000 | Qubit scalability |

## 🔧 Technical Details

### Quantum Feature Encoding

- **Z-Feature Map**: Single-qubit rotations, depth O(r), NISQ-friendly
- **ZZ-Feature Map**: Entangling gates, depth O(r·n), medium expressibility
- **Pauli-Feature Map**: Maximum expressivity, depth O(r·n²)

### Preprocessing Pipeline

1. **SelectKBest**: Mutual information criterion, select 2n top features
2. **MinMaxScaler**: Scale to [0, π] for quantum rotation gates
3. **PCA**: Reduce to exactly n components (matching qubit count)

### GPU Acceleration

- Qiskit Aer GPU-accelerated statevector simulator
- Single-precision floating point for memory efficiency
- Transpilation optimization level 2
- Multi-GPU parallel model evaluation

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{author2026quantum,
  title={Comprehensive Evaluation of Quantum Machine Learning for IoT Network Intrusion Detection: Novel Ensemble Methods and GPU-Accelerated Analysis},
  author={Author, First and Author, Second},
  booktitle={IEEE DCAS 2026},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [IoTID20](https://ieee-dataport.org/open-access/iot-network-intrusion-dataset) by Ullah & Mahmoud
- Quantum Framework: [Qiskit](https://qiskit.org/) by IBM
- This research was conducted using GPU-accelerated quantum simulation

## 📧 Contact

- **GitHub**: [@ocblvck](https://github.com/ocblvck)

---

<p align="center">
  <b>⚛️ Bridging Quantum Computing and IoT Security ⚛️</b>
</p>
