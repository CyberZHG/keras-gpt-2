# Keras GPT-2

[![Travis](https://travis-ci.org/CyberZHG/keras-gpt-2.svg)](https://travis-ci.org/CyberZHG/keras-gpt-2)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-gpt-2/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-gpt-2)
[![Version](https://img.shields.io/pypi/v/keras-gpt-2.svg)](https://pypi.org/project/keras-gpt-2/)
![Downloads](https://img.shields.io/pypi/dm/keras-gpt-2.svg)
![License](https://img.shields.io/pypi/l/keras-gpt-2.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-gpt-2/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-gpt-2/blob/master/README.md)\]

[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)预训练权重加载和预测。

## 安装

```bash
pip install keras-gpt-2
```

## 示例

```python
import os
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate


model_folder = 'xxx/yyy/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


print('从预训练checkpoint加载模型……')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('读取BPE词典……')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('生成文本……')
output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=1)

# 在使用117M模型且top_k设置为1的情况下，输出为：
# "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
print(output[0])
```
