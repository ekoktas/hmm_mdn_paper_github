# State Aware Traffic Generation for Real-Time Network Digital Twins

[![arXiv](https://img.shields.io/badge/arXiv-2509.12860-b31b1b.svg)](https://arxiv.org/abs/2509.12860)

This repository contains the official source code and implementation for the paper **"State Aware Traffic Generation for Real-Time Network Digital Twins"**, presented at the **2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)**.

**This work was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under grant number 16KIS2259 (SUSTAINET-inNOvAte).**

## Abstract

This project implements a hybrid machine learning approach combining **Hidden Markov Models (HMM)** and **Mixture Density Networks (MDN)** to generate high-fidelity synthetic network traffic. This approach is designed for Network Digital Twins, enabling the accurate modeling of state-dependent traffic characteristics (such as HTTP payload sizes and inter-arrival times) while maintaining temporal correlations.

## Repository Structure

- **`models/`**: Contains saved model checkpoints and weights.
- **`real_data/`**: Folder for input datasets (e.g., `df_raw_HTTP.csv`).
- **`synth_data/`**: Output folder where generated synthetic traffic is saved.
- **.ipynb**: Jupyter notebooks containing the training and generation logic are located in the root directory.

## Installation

To run this code, please ensure you have Python installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/ekoktas/hmm_mdn_paper_github.git
cd hmm_mdn_paper_github

# Install dependencies
pip install -r requirements.txt
```

## Usage

The implementation is provided via Jupyter Notebooks. To reproduce the results or train the models:

```bash
jupyter lab
# or
jupyter notebook
```

Open the notebooks in the file explorer to view the step-by-step implementation of the HMM-MDN training and traffic generation.

## Citation

If you use this code or the ideas presented in our paper for your research, please cite the following article:

```bibtex
@INPROCEEDINGS{11274598,
  author={Koktas, Enes and Rost, Peter},
  booktitle={2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)}, 
  title={State Aware Traffic Generation for Real-Time Network Digital Twins}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Hidden Markov models;Generators;Real-time systems;HTTP;Digital twins;Windows;Feeds;Servers;Mirrors;Payloads;hidden Markov model;mixture density network;mobile communication;network digital twin;synthetic traffic generation},
  doi={10.1109/PIMRC62392.2025.11274598}}
```