# Curiosity Mastcam Novelty Detection

Python 3.11.8 <- Elo Elo 3 2 0 


How to run?

Choose model from:
- GAN
- VAE 
- FLOW 

python main.py --model VAE --epochs 50 --lr 0.0005 --device cuda


Dane pochodzą z:

https://zenodo.org/records/3732485

umieścić odpakowanie w folderze dataset

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