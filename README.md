# Seq2Seq English-French Machine Translation Model Based on TensorFlow Framework

The content of this article is mainly based on English-French Parallel Corpus to implement a simple English-French translation model. The code framework uses TensorFlow 1.12.0.


## Seq2Seq model structure
![](pic.png)

## Usage method
1. python data_process.py
2. python seq2seq_en_fr_translation_model.py
3. python use_trained_model.py

## Usage model

```
【Input】
  Word Ids:      [120, 89, 131, 217, 171, 38, 180, 204, 208, 89, 103, 192, 213, 177]
  English Words: ['france', 'is', 'never', 'cold', 'during', 'september', ',', 'and', 'it', 'is', 'snowy', 'in', 'october', '.']

【Prediction】
  Word Ids:      [47, 235, 302, 325, 64, 204, 127, 81, 284, 117, 302, 72, 204, 63, 256, 1]
  French Words: ['la', 'france', 'est', 'jamais', 'froid', 'en', 'septembre', ',', 'et', 'il', 'est', 'neigeux', 'en', 'janvier', '.', '<EOS>']

【Full Sentence】
la france est jamais froid en septembre , et il est neigeux en janvier . <EOS>
```

## Code interpretation
> Code reference https://github.com/NELSONZHAO/zhihu/tree/master/machine_translation_seq2seq?1526990148498

[基于TensorFlow框架的Seq2Seq英法机器翻译模型](https://zhuanlan.zhihu.com/p/37148308)

[tensorflow Seq2seq编码函数详解](https://www.jianshu.com/p/9925171f692f)

[machine_translation_seq2seq.ipynb](https://nbviewer.jupyter.org/github/NELSONZHAO/zhihu/blob/master/machine_translation_seq2seq/machine_translation_seq2seq.ipynb)
