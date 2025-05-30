# QuEST: Stable Training of LLMs with 1-Bit Weights and Activations

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2502.05003)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=e3ClKTyVrkI)

<img width="795" alt="image" src="https://github.com/user-attachments/assets/bc615c35-fefb-4d51-b42f-59d5e07f2e46" />

QuEST is a Quantization-Aware Training (QAT) method that allows stable training of 1-bit Llama-type LLMs, and makes 4-bit training Pareto-optimal.

**[UPDATE 30.05.25]: QuEST has been accepted to ICML2025**

## Quickstart 

Create a conda environment and install dependencies (we recommend Python 3.10):

```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

Run a simple training on the SlimPajama 6B dataset:
```bash
python ./src/main.py
```

The above command trains a 213.34M parameters model with the Llama-style architecture. We recommend to use the `--compile` flag that speeds up training noticeably (up to 20% in our setup).

## Abstract

One approach to reducing the massive costs of large language models (LLMs) is the use of quantized or sparse representations for training or deployment. While post-training compression methods are very popular, the question of obtaining even more accurate compressed models by directly training over such representations, i.e., Quantization-Aware Training (QAT), is still open: for example, a recent study ([arXiv:2411.04330v2](https://arxiv.org/abs/2411.04330v2)) put the "optimal" bit-width at which models can be trained using QAT, while staying accuracy-competitive with standard FP16/BF16 precision, at 8-bits weights and activations.

We advance this state-of-the-art via a new method called QuEST, which is Pareto-competitive with FP16, i.e., it provides better accuracy at lower model size, while training models with weights and activations in 4-bits or less. Moreover, QuEST allows stable training with 1-bit weights and activations. QuEST achieves this by improving two key aspects of QAT methods: (1) accurate and fast quantization of the (continuous) distributions of weights and activations via Hadamard normalization and MSE-optimal fitting; (2) a new trust gradient estimator based on the idea of explicitly minimizing the error between the noisy gradient computed over quantized states and the "true" (but unknown) full-precision gradient. Experiments on Llama-type architectures show that QuEST induces stable scaling laws across the entire range of hardware-supported precisions, and can be extended to sparse representations. We provide GPU kernel support showing that models produced by QuEST can be executed efficiently. Our code is available at [this https URL](https://github.com/IST-DASLab/QuEST). 

## Quantization

See `train.sh` for an example on how to run quantized training.

## INT4 Inference Kernels

We provide Triton/CUDA kernels for INT4 Inference on NVIDIA Ampere GPUs. The code is particularly optimized for RTX 4090.

### Install

#### Dependencies

- cmake
- C++ compiler (GCC/clang/...)
- nvcc

#### Instructions

See `gemm-quest` [README](./gemm-quest/README.md).

## Testing the Models

This [notebook](src/HadamardTesting.ipynb) and similar provide a way to test that the trained model generate coherent text and are, indeed, serveable in quantized formats.

### HellaSWAG Benchmark

To run the HellaSWAG benchmark, please run the following command:

```bash
python ./src/eval_hswag.py --model_name "<MODEL_OUTPUT_NAME>" --ckpts_dir "./ckpts"
```

## Source Code

This repository is based on the [epfml/schedules-and-scaling](https://github.com/epfml/schedules-and-scaling) repository for their "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" paper. We thank the original creators for making public and open-licenced. 

## Cite This Work
```
@misc{panferov2025queststabletrainingllms,
      title={QuEST: Stable Training of LLMs with 1-Bit Weights and Activations}, 
      author={Andrei Panferov and Jiale Chen and Soroush Tabesh and Roberto L. Castro and Mahdi Nikdan and Dan Alistarh},
      year={2025},
      eprint={2502.05003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.05003}, 
}
```

