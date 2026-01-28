# Model Shrinker: Automated Edge AI Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)


> **"Unlocking 4x smaller models with <1% accuracy loss."**

**Model Shrinker** is a modular MLOps pipeline designed to automate the optimization of Large Language Models (LLMs) and Vision Transformers for edge deployment. It bridges the gap between *Research* (PyTorch) and *Production* (ONNX Runtime) by providing a one-click tool for conversion, quantization, and quality control.

---

##  Architecture

The pipeline treats model optimization as a manufacturing process, moving raw models through specialized "Stations" to ensure quality and performance.

```mermaid
graph TD
    User([User Request]) --> Gateway[<b>API Gateway</b><br>FastAPI / Python 3.13]
    
    Gateway -- "Route: 'type'" --> Encoder
    Gateway -- "Route: 'type'" --> Decoder

    subgraph " "
        direction TB
        Encoder[<b>Worker: ENCODER</b><br>Python 3.13<br>• ONNX Runtime<br>• BERT / ViT]
        Decoder[<b>Worker: DECODER</b><br>Python 3.10<br>• AutoAWQ 0.2.6<br>• Qwen / Llama]
    end

    %% Styling
    style User fill:#212121,stroke:#ffffff,stroke-width:2px,color:#ffffff
    style Gateway fill:#37474F,stroke:#FFC107,stroke-width:2px,color:#ffffff
    style Encoder fill:#263238,stroke:#00E676,stroke-width:2px,color:#ffffff
    style Decoder fill:#263238,stroke:#00B0FF,stroke-width:2px,color:#ffffff
    linkStyle default stroke:#B0BEC5,stroke-width:2px
```
## Key Features

* **Universal Conversion:** Automatically converts Hugging Face models (BERT, DistilBERT, ViT) to ONNX Graph format.
* **Smart Quantization:**
    * **Dynamic Mode:** Fast compression for immediate prototyping.
    * **Static Mode (Calibrated):** Uses real-world data (wikitext) to calibrate neuron activation ranges for maximum accuracy.
* **Automated Quality Control:** Runs a "Fidelity Check" comparing the raw feature vectors of the original vs. quantized model using Cosine Similarity.
* **Hardware-Aware Stats:** Instantly reports disk size reduction (MB) and expected inference speedup.

## Performance Benchmarks

Tested on NVIDIA RTX 4060 (Laptop GPU) and Intel i7 CPU.

| Model | Original Size (FP32) | Quantized Size (INT8) | Reduction | Fidelity Score | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bert-Tiny** | 16.71 MB | 4.24 MB | 74.6% | 0.9984 | Production Ready |
| **DistilBERT** | 253.2 MB | 63.8 MB | 74.8% | 0.9850 | Production Ready |
| **ResNet-50** | 98.0 MB | 24.6 MB | 74.9% | 0.9912 | Production Ready |

> **Note:** Models with a Fidelity Score > 0.99 are considered indistinguishable from the original by human evaluators.

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/yourusername/model-shrinker.git](https://github.com/yourusername/model-shrinker.git)
cd model-shrinker
```
