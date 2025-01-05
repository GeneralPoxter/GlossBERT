# GlossBERT

This code implements the paper "GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge" ([Huang et al. 2019](https://aclanthology.org/D19-1355/)).
We use the Hugging Face `transformers` library to train GlossBERT on a 200k sample of the SemCor 3.0 dataset.

## Overview

GlossBERT is a method for word sense disambiguation (WSD) that trains a BERT-based sequence classifier to determine whether a given context-gloss pair represents a correct word sense.
It adapts the WSD task for large language models by reframing it as a binary classification problem.
For each word $w$ and candidate sense $s$ of $w$, we construct the context-gloss pair:

```
[CLS] <context of w> [SEP] <gloss of s> [SEP]
```

GlossBERT classifies a pair as either 1 ($s$ is the correct sense for $w$ in the context) or 0 ($s$ is not the correct sense).
Inference for determining the sense of a word $w$ is performed by classifying each context-gloss pair constructed from $w$ and one of its candidate senses.
We select the sense with the highest probability for class 1 to be the predicted sense.

We also extend GlossBERT by proposing GlossBERT-C, a model for *contrastive* context-gloss classification.
This extension is motivated by the hypothesis that BERT might learn more meaningful differences between senses if positive and negative examples are juxtaposed in the same input sequence during training and inference.

To make the task contrastive, we train BERT for binary *triple* (as opposed to pair) classification, where each triple consists of a context and two glosses to be compared.
Formally, given a word $w$ and two senses $s_1$ and $s_2$, we construct the context-gloss triple:

```
[CLS] <context of w> [SEP] <gloss of s1> [SEP] <gloss of s2> [SEP]
```

GlossBERT-C classifies a triple as either 0 ($s_1$ is a better sense for $w$ than $s_2$) or 1 ($s_2$ is a better sense than $s_1$).
Inference for determining the sense of a word $w$ is performed by classifying all triples formed from all possible pairs of candidate senses $(s_1, s_2)$ where $s_1\neq s_2$.
The sense that is classified as ``better'' in the most number of triples is selected as the predicted sense.
Compared to GlossBERT, inference cost scales from $O(n)$ to $O(n^2)$, where $n$ is the number of WordNet senses for $w$.

## Installation

Create a Python 3.11 environment and install packages:
```
pip install -r requirements.txt
```

Download the WordNet corpus in Python:
```python
import nltk
nltk.download('wordnet')
```

## Usage
It is **strongly recommended** that training and evaluation be performed on a machine with at least a 10GB GPU.

### From scratch
To locally train and then evaluate GlossBERT-200k:
```
python gloss_bert.py --mode train
python gloss_bert.py --mode eval
```
and for GlossBERT-C-200k:
```
python gloss_bert_contrastive.py --mode train
python gloss_bert_contrastive.py --mode eval
```

### From checkpoints
To evaluate the trained checkpoint for GlossBERT-200k:
```
python gloss_bert.py --mode eval --model-name GeneralPoxter/GlossBERT-200k
```
and for GlossBERT-C-200k:
```
python gloss_bert_contrastive.py --mode eval --model-name GeneralPoxter/GlossBERT-contrastive-200k
```

To evaluate the checkpoint from the original paper's authors:
```
python gloss_bert.py --mode eval --model-name GeneralPoxter/GlossBERT-original
```

## Results

For GlossBERT and GlossBERT-C, we train a model checkpoint on both 100k and 200k context-gloss samples, and evaluate on a fixed holdout of 10k words.

### GlossBERT
| Model               | Accuracy |
| ------------------- | -------- |
| BERT<sub>BASE</sub> | 28.04%   |
| Huang et al. 2019   | 62.12%   |
| GlossBERT-100k      | 66.96%   |
| GlossBERT-200k      | 69.27%   |

### GlossBERT-C
| Model               | Accuracy |
| ------------------- | -------- |
| BERT<sub>BASE</sub> | 57.01%   |
| GlossBERT-C-100k    | 68.72%   |
| GlossBERT-C-200k    | 70.10%   |