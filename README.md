# Seq2Seq English-French Machine Translation Model Based on TensorFlow Framework

The content of this article is mainly based on English-French Parallel Corpus to implement a simple English-French translation model. The code framework uses TensorFlow 1.12.0.


为Seq2Se写的
## 首先是`data_process.py`:<br>
读入源端和目标端文件
```
with open(os.path.join("data","small_vocab_en"), "r", encoding="utf-8") as f:
    source_text = f.read()
with open(os.path.join("data","small_vocab_fr"), "r", encoding="utf-8") as f:
    target_text = f.read()
```
接下来，按照换行符将原始文本分割成句子,统计句子的数量，最大句子长度和平均长度，word_counts保存的就是每个句子单词数
```
sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]

sentences = target_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
```
构造英文词典\法文词典，以及对应的id映射
```
source_vocab = list(set(source_text.lower().split()))

target_vocab = list(set(target_text.lower().split()))

# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符
# 构造英文映射字典
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}
# 构造法语映射词典
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}
```

得到了映射词典之后就需要对source和target的内容进行映射
```
# 对源句子进行转换 Tx = max_source_sentence_length
source_text_to_int = []
for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int,
                                          max_source_sentence_length,
                                          is_target=False))
# 对目标句子进行转换  Ty = max_target_sentence_length
target_text_to_int = []
for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int,
                                          max_target_sentence_length,
                                          is_target=True))
```

其中text_to_int()函数如下：需要注意一点的是，不需要这么早进行padding，因为需要在get_batches()统计一下每个句子词的真实长度，<br>
以便用于tf.nn.dynamic_rnn和masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

```
def text_to_int(sentence, map_dict, is_target=False):
    # 用<PAD>填充整个序列
    text_to_idx = []
    # unk index
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果是输入源文本,如果指定键的值不存在时，返回unk对应的id
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 否则，对于输出目标文本需要做<EOS>的填充最后
    else:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    return text_to_idx
```

## 接下来是`attention.py`:<br>
具体可以见下图笔记：<br>

这个代码主要在translation_model.py下面改的，主要就是融入了attention模型

```
# *****************************Attention********************************
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output)
    # decoder_cell 得到的是Si
    '''
    attention_layer_size: 用来控制我们最后生成的attention是怎么得来的，如果是None，则直接返回对应attention mechanism计算得到的加权和向量；
    如果不是None，则在调用_compute_attention方法时，得到的加权和向量还会与output进行concat，
    然后再经过一个线性映射，变成维度为attention_layer_size的向量
    AttentionWrapperState 用来存储整个计算过程中的 state，和 RNN 中的 state 类似，只不过这里额外还存储了 attention、time 等信息。 
    AttentionWrapper 主要用于封装 RNNCell，继承自 RNNCell，封装后依然是 RNNCell 的实例，可以构建一个带有 Attention 机制的 Decoder。 

    '''
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                       attention_layer_size=rnn_size)
    # 得到一个全0的初始状态，[b, rnn_size]
    de_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
    # 做一个变换，最后一个维度为target_vocab_size
    out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)

```
在下代码中，主要改变的就是`de_state`以及`out_cell`两个参数，分别表示初始状态以及经过attention的cell单元：
```
    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(de_state,
                                              out_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)
    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(de_state,
                                               out_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["<GO>"],
                                               target_vocab_to_int["<EOS>"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)
```


## Code interpretation
> Code reference https://github.com/NELSONZHAO/zhihu/tree/master/machine_translation_seq2seq?1526990148498

[基于TensorFlow框架的Seq2Seq英法机器翻译模型](https://zhuanlan.zhihu.com/p/37148308)

[tensorflow Seq2seq编码函数详解](https://www.jianshu.com/p/9925171f692f)

[machine_translation_seq2seq.ipynb](https://nbviewer.jupyter.org/github/NELSONZHAO/zhihu/blob/master/machine_translation_seq2seq/machine_translation_seq2seq.ipynb)
