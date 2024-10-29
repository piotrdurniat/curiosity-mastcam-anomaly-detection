# Curiosity Mastcam Novelty Detection

This repository contains the code for the novelty detection project using the Curiosity Mastcam images.

## Report

- [Full report PDF](./report/report.pdf)

## Requirements

- Python 3.11+

Install the required packages by running:

```bash
pip install -r requirements_linux_cpu.txt
```
or 
```bash
pip install -r requirements_linux_gpu.txt
```

## Usage

### Train

Example usage (run from the root directory):

```bash
python3 main.py --lr 1e-6 --model VAE --epochs 1 --device cuda
```

Arguments:
- `--model` - model to use (one of GAN, VAE, FLOW)
- `--epochs` - number of epochs to train
- `--lr` - learning rate
- `--device` - device to use (eg.cpu or cuda)

### Test

Example usage (run from the root directory):

```bash
python3 ./main_test.py --model VAE --batch 1 --device cpu --load ./vae/model/vae_model-250-epoch.pth --save results/vae
```

Arguments:
- `--model` - model to use (one of GAN, VAE, FLOW)
- `--batch` - batch size
- `--device` - device to use (eg.cpu or cuda)
- `--load` - path to the model to load
- `--save` - path to save the results

## Dataset

Multispectral images of Mars taken by the Curiosity rover. The dataset is divided into four parts: train_typical, validation_typical, test_typical, and test_novel.

Source:
- https://zenodo.org/records/3732485

To download the dataset run:

```bash
cd dataset
curl -O https://zenodo.org/records/3732485/files/test_novel.zip
curl -O https://zenodo.org/records/3732485/files/test_typical.zip
curl -O https://zenodo.org/records/3732485/files/train_typical.zip
curl -O https://zenodo.org/records/3732485/files/validation_typical.zip
unzip test_novel.zip
unzip test_typical.zip
unzip train_typical.zip
unzip validation_typical.zip
```
