# NBCR-ac4C

N4-acetylcytidine (ac4C) plays a crucial role in regulating cellular biological processes, particularly in gene expression regulation and disease development.

we employ Nucleotide Transformer and DNABERT2 to construct contextual embedding of nucleotide sequences, which effectively mines and express context relations between different features in the sequence. CNN and ResNet18 are then applied to further extract shallow and deep knowledge from the context embedding. we propose a deep learning approach called NBCR-ac4C based on pre-trained models.

The source code and datasets(both training and testing datasets) can be freely download from the github

# Environment requirements
Before running, please make sure the following packages are installed in Python environment:

einops==0.8.0
gensim==4.2.0
mamba_ssm==2.1.0
mmcv==2.1.0
mmengine==0.10.4
numpy==1.21.6
pandas==1.3.5
scikit_learn==1.0.2
timm==0.9.12
torch==1.13.1
torchvision==0.14.1
tqdm==4.65.2
transformers==4.30.2


# RUN
Changing working dir to NBCR, python resnet.py


# link

Nucleotide transformer can be found https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species

DNABERT2 can be found https://huggingface.co/zhihan1996/DNABERT-2-117M

the model pth can be found [https://huggingface.co/hanyucold/NBCR](https://huggingface.co/hanyucold/NBCR/tree/main)
![image](https://github.com/2103374200/NBCR/assets/60246005/02b650d9-0d81-4fc2-88e9-efa7de981d9d)


