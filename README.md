# GlossBERT

This code implements the paper "GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge" ([Huang et al. 2019](https://aclanthology.org/D19-1355/)).
We use the Hugging Face `transformers` library to train GlossBERT on a 200k sample of the SemCor 3.0 dataset.

## Overview

TODO

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