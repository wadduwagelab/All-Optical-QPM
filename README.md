# Differentiable Microscopy Designs an All Optical Phase Retrieval Microscope


We introduce Differentiable Microscopy ($\partial \mu$), a deep learning-based top-down design paradigm for optical microscopes. Using all-optical phase retrieval as an illustrative example, we demonstrate the effectiveness of data-driven microscopy design through $\partial \mu$. Furthermore, we conduct comprehensive comparisons with competing methods, showcasing the consistent superiority of our learned designs across multiple datasets, including biological samples. This repository contains the official PyTorch implementation of our designs. 



## Running Pretrained Models

Here we provide the Google Colab notebooks to reproduce the results of our best performing models for each dataset for our microscopy designs.

- <b>Expected results of each notebook :</b>
  - SSIM score for the unseen dataset. SSIM values should match the below table.
  - Groundruth and the corresponding reconstructed images will be printed in the notebooks
- <b>Expected runtime for Setting up and Inference on Colab</b> : ~2 Minutes (Make sure to select a GPU runtime)

### Performance of Optical Models (SSIM)

| Model                            | MNIST  | HeLa [0,Pi] | HeLa [0,2Pi] | Bacteria | Colab Notebook |
|----------------------------------|--------|-------------|--------------|----------|----------------|
| Generalized Phase Contrast (GPC) | 0.5134 | 0.5652      | 0.4056       | 0.6740   | <a href="https://colab.research.google.com/github/Bantami/All-Optical-QPM/blob/main/Colab/GPC_baseline_inference_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                |
| Learnable Fourier Filter (LFF)   | 0.9184 | 0.7217      | 0.5921       | 0.9820   | <a href="https://colab.research.google.com/github/Bantami/All-Optical-QPM/blob/main/Colab/lff_pretrained_model_inference_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               |
| PhaseD2NN                        | 0.9146 | 0.6254      | 0.4854       | 0.9915   |    <a href="https://colab.research.google.com/github/Bantami/All-Optical-QPM/blob/main/Colab/d2nn_pretrained_model_inference_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>            |

### Performance of complex-valued linear CNNs (SSIM)
| Model                     | MNIST  | HeLa [0,Pi] | HeLa [0,2Pi] | Bacteria | Colab Notebook |
|---------------------------|--------|-------------|--------------|----------|----------------|
| Complex-valued linear CNN | 0.9727 | 0.9052       | 0.7059       | 0.9660   | <a href="https://colab.research.google.com/github/Bantami/All-Optical-QPM/blob/main/Colab/cnn_inference_colab.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>               |



## Training Models

### 1. Clone the reposoitory

```bash
git clone https://github.com/Bantami/All-Optical-QPM.git
```

### 2. Download HeLa and Bacteria datasets and set data_dir path in `dataloaders.py`

[HeLa dataset link](https://drive.google.com/uc?id=1ickDfs6bA-YM7RQSaMPRqFnC7YApjW8e), [Bacteria dataset link](https://drive.google.com/uc?id=1CRaPVYVUs-vJA6SoXqeNhBRiqMwITbv6)

To download the dataset through the commandline, follow the below steps

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1ickDfs6bA-YM7RQSaMPRqFnC7YApjW8e
gdown https://drive.google.com/uc?id=1CRaPVYVUs-vJA6SoXqeNhBRiqMwITbv6

mkdir datasets/
unzip -qq hela.zip -d datasets/

mkdir -p datasets/bacteria/val
unzip -j -qq bacteria.zip -d datasets/bacteria/val

```

Update `dataloaders.py` to set,
- MNIST `data_dir` (any existing path will work) in line [30](https://github.com/Bantami/All-Optical-QPM/blob/683d3db9c9fee2cfd3c0545c26dc2c07ba019669/modules/dataloaders.py#L30)
and 
- HeLa `data_dir` in line [57](https://github.com/Bantami/All-Optical-QPM/blob/683d3db9c9fee2cfd3c0545c26dc2c07ba019669/modules/dataloaders.py#L57)
- Bacteria `data_dir` in line [111](https://github.com/Bantami/All-Optical-QPM/blob/3b872b2dd8bc9c9ac2218a83db487d738fd83bb2/modules/dataloaders.py#L111)






### 3. Setting up a new environment and adding the kernel to Jupyter

* This should take only few minutes

```bash
## create new environment
conda create -n qpm_env python=3.6
source activate qpm_env

## Adding new environment to JupyterLab
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=qpm_env

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c conda-forge matplotlib
conda install -c conda-forge wandb

#install remaining packages through pip
pip install -r requirements.txt
```

### 4. Training the Optical/Baseline Models for Different Datasets


Please find the training notebooks for each of the models for each dataset below.

> run notebook_name.ipynb after selecting the newly added kernal (qpm_env)

> results will be saved in the folder "results" which will be created in the parent directory w.r.t to where the notebook is located.

- Expected results of each training notebook (in `results` folder):
  - Saved model for latest epoch
  - Loss curves figure
  - Input/Reconstructed images comparison figure for each epoch (SSIM, L1, BerHu loss will be displayed)


#### Learnable Fourier Filter (LFF)
- [LFF - MNIST](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/LearnableFourierFilter/LFF_MNIST.ipynb)
- [LFF - HeLa [0,Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/LearnableFourierFilter/LFF_HeLA_pi.ipynb)
- [LFF - HeLa [0,2Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/LearnableFourierFilter/LFF_HeLA.ipynb)

#### PhaseD2NN  
- [PhaseD2NN - MNIST](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/PhaseD2NN/PhaseD2NN_mnist.ipynb)
- [PhaseD2NN - HeLa [0,Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/PhaseD2NN/PhaseD2NN_hela_pi.ipynb) 
- [PhaseD2NN - HeLa [0,2Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/PhaseD2NN/PhaseD2NN_hela_2pi.ipynb)

#### Complex-valued linear CNN
- [Complex-CNN - MNIST](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/ComplexCNN/complexCNN_MNIST.ipynb)
- [Complex-CNN - HeLa [0,Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/ComplexCNN/complexCNN_HeLA_pi.ipynb)
- [Complex-CNN - HeLa [0,2Pi]](https://github.com/Bantami/All-Optical-QPM/blob/main/Notebooks/ComplexCNN/complexCNN_HeLA.ipynb)


## Directory Structure:

```

├── Colab
│   ├── cnn_inference_colab.ipynb
│   ├── d2nn_pretrained_model_inference_colab.ipynb
│   ├── GPC_baseline_inference_colab.ipynb
│   └── lff_pretrained_model_inference_colab.ipynb
├── colab_setup.sh
├── modules
│   ├── d2nn_layers.py
│   ├── d2nn_models.py
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── diffraction.py
│   ├── eval_metrics.py
│   ├── fourier_model.py
│   ├── loss.py
│   ├── other_models.py
│   ├── train.py
│   ├── train_utils.py
│   └── vis_utils.py
├── Notebooks
│   ├── ComplexCNN
│   │   ├── complexCNN_HeLA.ipynb
│   │   ├── complexCNN_HeLA_pi.ipynb
│   │   └── complexCNN_MNIST.ipynb
│   ├── GPC.ipynb
│   ├── LearnableFourierFilter
│   │   ├── LFF_HeLA.ipynb
│   │   ├── LFF_HeLA_pi.ipynb
│   │   └── LFF_MNIST.ipynb
│   ├── PhaseD2NN
│   │   ├── PhaseD2NN_hela_2pi.ipynb
│   │   ├── PhaseD2NN_hela_pi.ipynb
│   │   └── PhaseD2NN_mnist.ipynb
│   └── results
├── overview.png
├── README.md
└── requirements.txt


```
