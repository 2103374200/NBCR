# NBCR-ac4C

N4-acetylcytidine (ac4C) plays a crucial role in regulating cellular biological processes, particularly in gene expression regulation and disease development.

we employ Nucleotide Transformer and DNABERT2 to construct contextual embedding of nucleotide sequences, which effectively mines and express context relations between different features in the sequence. CNN and ResNet18 are then applied to further extract shallow and deep knowledge from the context embedding. we propose a deep learning approach called NBCR-ac4C based on pre-trained models.

The source code and datasets(both training and testing datasets) can be freely download from the github

# Environment requirements
Before running, please make sure the following packages are installed in Python environment:

python  3.7.16

Pillow	9.5.0

gensim	4.2.0

h5py	3.8.0

numpy	1.21.6

torch	1.13.1

torchvision	0.14.1

# RUN
Changing working dir to NBCR, python resnet.py


the model pth can be found [https://huggingface.co/hanyucold/NBCR](https://huggingface.co/hanyucold/NBCR/tree/main)
![image](https://github.com/2103374200/NBCR/assets/60246005/02b650d9-0d81-4fc2-88e9-efa7de981d9d)


