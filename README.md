# MeGPT: Medical image report generation integrating large language models

## Introduction

With the rapid development of artificial intelligence technology, the automatic generation of medical imaging reports has become one of the key technologies to improve medical efficiency. However, the existing methods still have deficiencies in image feature extraction, cross-modal alignment and generation accuracy. Therefore, this paper proposes a Medical image report generation model Medical-Enhanced Generative Pre-trained Transformer(MeGPT) that integrates large language models. Through multi-module collaborative optimization, the generation quality of the report has been significantly improved. Firstly, aiming at the problem of low fine-grained medical images, the Feature Enhancement Module (FEM) and dynamic over-parameterized convolution (DOConv) were designed. Through multi-branch feature extraction and fusion, the discriminative power of key pathological features was enhanced. Secondly, a dynamic Prompt mechanism is proposed. Based on the image content, the optimal prompt is adaptively selected to guide the large language model to generate more accurate text descriptions. Furthermore, the L2 regularization optimization loss function is introduced to effectively suppress overfitting and enhance the generalization ability of the model. The experiment designed comparative experiments and ablation experiments based on the IU X-Ray dataset. MeGPT outperformed mainstream models (such as VLCI+TISR, R2GenCMN, etc.) in indicators such as BLEU-1 (0.487) and ROUGE (0.372). The ablation experiments verified the effectiveness of each module: the feature enhancement module improved the local feature matching ability, the dynamic prompt mechanism enhanced the quality of text generation, and L2 regularization significantly improved the robustness of the model. This research provides an efficient solution for the automation of medical imaging reports. Our innovation points include: (1) Feature enhancement methods for low-fine-grained images; (2) Dynamic prompt selection strategy based on clustering (3) The lightweight architecture design takes into account both performance and computing efficiency.

## Getting Started

### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
cd R2GenGPT
pip install -r requirements.txt
```

**2. Prepare the training dataset**

IU-xray: download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)

Mimic-cxr: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training

For shallow alignment

```bash
bash scripts/4-1.shallow_run.sh
```

For delta alignment

```bash
bash scripts/5-1.delta_run.sh
```

For deep alignment

```bash
bash scripts/6-1.deep_run.sh
```

### Testing (For MIMIC-CXR)

You can download our pretrained Delta checkpoints for [Here](https://drive.google.com/drive/folders/1ywEITWfYIAAYy0VY1IZ24Ec_GoNmkqIY?usp=sharing)

For shallow alignment

```bash
bash scripts/4-2.shallow_test.sh
```

For delta alignment

```bash
bash scripts/5-2.delta_test.sh
```

For deep alignment

```bash
bash scripts/6-2.shallow_test.sh
```


## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.

