**Adversarial Examples for Extreme Multilabel Text Classification [1]**

The code is adapted from the source codes of BERT-ATTACK [2], APLC_XLNet [3], and AttentionXML [4]

## Requirements
The code has been tested by the following packages:

* python==3.6.13
* boto3==1.17.70
* ruamel.yaml==0.16.12
* numpy==1.19.2
* scipy==1.5.4
* matplotlib==3.2.2
* scikit-learn==0.24.2
* transformers==2.9.0
* torch==1.4.0
* nltk==3.4
* pandas==1.1.5
* requests==2.25.1
* tqdm==4.60.0

## A small experiment on Wikipedia-31K with only 10 samples per bin

Downolad the data and the APLC_XLNet model trained on this data as follows:
```bash
bash download_data_model.sh
```

For preprocessing the data, run the following:
```bash
bash preprocess.sh
```

To run positive-targeted attacks with 10 samples per bin, run the following:
```bash
bash pos_attack.sh
```

To check the results of the attacks, run the following:
```bash
bash results.sh
```


## References
[1] Qaraei and Babbar, [Adversarial Examples for Extreme Multilabel Text Classification](https://arxiv.org/pdf/2112.07512.pdf), Arxiv 2021

[2] Li et al., [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://arxiv.org/abs/2004.09984), EMNLP 2020

[3] Ye et al., [Pretrained Generalized Autoregressive Model with Adaptive Probabilistic Label Clusters for Extreme Multi-label Text Classification](http://arxiv.org/abs/2007.02439), ICML 2020

[4] You et al., [AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727), NeurIPS 2019
