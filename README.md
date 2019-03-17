# Language Modelling

## Transfer Learning in Langage Modelling (Classification)
Implementation of [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (Fast.ai). It's used as default now.

[Blog on ULMFit at Intel IDZ](https://software.intel.com/en-us/articles/transfer-learning-in-natural-language-processing)

## Transformer Support has been added
Prerequisites:
- [Tensorop](https://github.com/prajjwal1/tensorop)
For Installation, visit [docs for transformers]((https://github.com/prajjwal1/tensorop/docs/transformers.md))

Usage:

```
from tensorop import *
from tensorop.nlp import *
lm = Language_Model('gpt2')
lm.run_model()

## It will ask for the prompt
```
Support for Transformer XL,BERT is WIP

### Requirements:
- Pytorch
- Numpy
- Python 3.x
- Fast.ai lib

### Acquire the Repo
```shell
$ git clone https://github.com/prajjwal1/language-modelling
```

### Contributions
Contributions are always welcome in the form of pull requests with explanatory comments.
