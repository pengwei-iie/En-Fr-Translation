import tensorflow as tf
import numpy as np
import os
import pickle
from nltk.translate.bleu_score import sentence_bleu
# Number of Epochs
epochs = 10
# Batch Size
# batch_size = 16
batch_size = 256
# RNN Size
rnn_size = 128
# Number of Layers
rnn_num_layers = 2
# Embedding Size
encoder_embedding_size = 100
decoder_embedding_size = 125
# Learning Rate
lr = 0.001
# 每display_step轮打一次结果
display_step = 100
max_source_sentence_length = 20 # data_process里面计算过
max_target_sentence_length = 25

# 加载数据
prepared_data = np.load(os.path.join("preparing_resources","prepared_data.npz"))
# prepared_data = np.load(os.path.join("preparing_resources","prepared_data_cn.npz"))
source_text_to_int = prepared_data['X']
target_text_to_int = prepared_data['Y']
print("\nDATA shape:")
print("source_text_to_int_shape:\t", source_text_to_int.shape)
print("target_text_to_int_shape:\t", target_text_to_int.shape)
# 加载字典
with open(os.path.join("preparing_resources", "enfa_vocab_to_int.pickle"), 'rb') as f:
    source_vocab_to_int = pickle.load(f)
# 目标是法语的
with open(os.path.join("preparing_resources", "fa_vocab_to_int.pickle"), 'rb') as f:
# with open(os.path.join("preparing_resources", "cn_vocab_to_int.pickle"), 'rb') as f:
    target_vocab_to_int = pickle.load(f)

# 加载字到ID  然后得到ID到字，只是转换一下，没有人用
source_int_to_vocab = {idx: word for word, idx in source_vocab_to_int.items()}
target_int_to_vocab = {idx: word for word, idx in target_vocab_to_int.items()}
print("The size of English Map is : {}".format(len(source_vocab_to_int)))
print("The size of French Map is : {}".format(len(target_vocab_to_int)))
# print("The size of Chinese Map is : {}".format(len(target_vocab_to_int)))

def pad_batch_sentence(batch, pad_id):
    max_length = max([len(sentence) for sentence in batch])
    sen = []
    for sentence in batch:
        sen.append(sentence + [pad_id] * (max_length - len(sentence)))
    return sen, max_length

def get_batches(sources, targets, batch_size):
    """
    获取batch
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = np.array(sources[start_i:start_i + batch_size])
        targets_batch = np.array(targets[start_i:start_i + batch_size])

        pad_idx = source_vocab_to_int.get("<PAD>")
        sources_batch_pad, max_source_len = np.array(pad_batch_sentence(sources_batch, pad_idx))

        targets_batch_pad, max_target_len = np.array(pad_batch_sentence(targets_batch, pad_idx))
        # max_target_sequence_len = tf.convert_to_tensor(max_target_sequence_len, dtype=tf.int32)
        # Need the lengths for the _lengths parameters
        # 不应该是对pad过的batch做长度的计算，因为都是25
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield sources_batch_pad, targets_batch_pad, source_lengths, targets_lengths, max_target_len



def model_inputs():
    """
    构造输入
    返回：inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len，类型为tensor
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, name="max_target_sequence_len")
    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len


def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
    """
    构造Encoder端

    @param rnn_inputs: rnn的输入
    @param rnn_size: rnn的隐层结点数
    @param rnn_num_layers: rnn的堆叠层数
    @param source_sequence_len: 英文句子序列的长度
    @param source_vocab_size: 英文词典的大小
    @param encoder_embedding_size: Encoder层中对单词进行词向量嵌入后的维度
    """
    # 对输入的单词进行词向量嵌入
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    # LSTM单元
    def get_lstm(rnn_size):
        lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    # 堆叠rnn_num_layers层LSTM
    lstms = tf.nn.rnn_cell.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
                                                        dtype=tf.float32)

    return encoder_outputs, encoder_states


def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):
    """
    对Decoder端的输入进行处理

    @param target_data: 法语数据的tensor
    @param target_vocab_to_int: 法语数据的词典到索引的映射
    @param batch_size: batch size
    """

    # 去掉batch中每个序列句子的最后一个单词。    第一维，从0到batch(开区间)。第二维，从0到倒数第一个(开区间)，步长都为1
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # 在batch中每个序列句子的前面添加”<GO>"
    decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["<GO>"]),
                                ending], 1)
    # for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
    #         get_batches(source_text_to_int, target_text_to_int, batch_size)):
    #     with tf.Session(graph=train_graph) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run([ending[1], decoder_inputs[1]], {inputs: source_batch, targets: target_batch, learning_rate: lr, source_sequence_len: sources_lengths, target_sequence_len: targets_lengths}))

    return decoder_inputs


def decoder_layer_train(zero_states, decoder_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer):
    """
    Decoder端的训练
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param target_sequence_len: 法语文本的长度,是一个list，指的是当前batch中每个序列的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    """
    # inputs的shape就是[batch_size, sequence_length, embedding_size] ，time_major=True时，inputs的shape为[sequence_length, batch_size, embedding_size]
    # 生成helper对象, 训练的时候，data是给定的，所以embeding是经过查表后的。而测试的时候，传入的是词向量表，然后再查表
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                        sequence_length=target_sequence_len,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                       training_helper,
                                                       zero_states,
                                                       output_layer)
    # 为真时会拷贝最后一个时刻的状态并将输出置零，程序运行更稳定，使最终状态和输出具有正确的值，在反向传播时忽略最后一个完成步。但是会降低程序运行速度。
    training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True)
    # training_decoder_outputs, final_state, final_sequence_lengths
    # for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
    #         get_batches(source_text_to_int, target_text_to_int, batch_size)):
    #     with tf.Session(graph=train_graph) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run([training_decoder_outputs.sample_id[0]], {inputs: source_batch, targets: target_batch, learning_rate: lr, source_sequence_len: sources_lengths, target_sequence_len: targets_lengths}))

    return training_decoder_outputs


def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size):
    """
    Decoder端的预测/推断

    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param start_id: 句子起始单词的token id， 即"<GO>"的编码
    @param end_id: 句子结束的token id，即"<EOS>"的编码
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    @batch_size: batch size
    """
    # 张量扩展，扩展为[batch_size]维度
    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")
    # start_tokens是一个向量 ：batch中每个序列起始输入的token_id
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
                                                                start_tokens,
                                                                end_id)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_states,
                                                        output_layer)

    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                        impute_finished=True)

    return inference_decoder_outputs


def decoder_layer(encoder_output, encoder_states, decoder_inputs, target_sequence_len,
                  max_target_sequence_len, rnn_size, rnn_num_layers,
                  target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size):
    """
    构造Decoder端

    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_inputs: Decoder端的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    @param target_vocab_size: 法语词典的大小
    @param decoder_embedding_size: Decoder端词向量嵌入的大小
    @param batch_size: batch size
    """

    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    def get_lstm(rnn_size):
        lstm = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm

    decoder_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])

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

    # output_layer logits 输出的是一个概率分布，词表大小
    output_layer = tf.layers.Dense(target_vocab_size)

    # 未加Atten之前输入的是decoder_cell， 加了之后输入的out_cell
    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(de_state,
                                              out_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)
    # training_logits(22) decoder_inputs(23)
    # for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
    #         get_batches(source_text_to_int, target_text_to_int, batch_size)):
    #     with tf.Session(graph=train_graph) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run([training_logits.sample_id[0]], {inputs: source_batch, targets: target_batch, learning_rate: lr, source_sequence_len: sources_lengths, target_sequence_len: targets_lengths}))

    # 共享参数
    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(de_state,
                                               out_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["<GO>"],
                                               target_vocab_to_int["<EOS>"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, batch_size,
                  source_sequence_len, target_sequence_len, max_target_sentence_len,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embeding_size,
                  rnn_size, rnn_num_layers, target_vocab_to_int):
    """
    构造Seq2Seq模型

    @param input_data: tensor of input data
    @param target_data: tensor of target data
    @param batch_size: batch size
    @param source_sequence_len: 英文语料的长度
    @param target_sequence_len: 法语语料的长度
    @param max_target_sentence_len: 法语的最大句子长度
    @param source_vocab_size: 英文词典的大小
    @param target_vocab_size: 法语词典的大小
    @param encoder_embedding_size: Encoder端词嵌入向量大小
    @param decoder_embedding_size: Decoder端词嵌入向量大小
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    """
    # 得到Encoder的输出和最后一个时刻的状态
    encoder_output, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)
    # 将最后一个步去掉，加入<Go>
    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_output,
                                                                        encoder_states,
                                                                        decoder_inputs,
                                                                        target_sequence_len,
                                                                        max_target_sentence_len,
                                                                        rnn_size,
                                                                        rnn_num_layers,
                                                                        target_vocab_to_int,
                                                                        target_vocab_size,
                                                                        decoder_embeding_size,
                                                                        batch_size)
    # training_decoder_outputs的seq长度变成了22
    # for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
    #         get_batches(source_text_to_int, target_text_to_int, batch_size)):
    #     with tf.Session(graph=train_graph) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         print(sess.run(training_decoder_outputs.sample_id[1], {inputs: source_batch, targets: target_batch, learning_rate: lr, source_sequence_len: sources_lengths, target_sequence_len: targets_lengths}))

    return training_decoder_outputs, inference_decoder_outputs

train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len = model_inputs()

    # max_target_sequence_len = max_target_sentence_length # axis = -1最后一个维度，batch_size * num_step--对列进行反转
    # max_target_sequence_len = 50  # axis = -1最后一个维度，batch_size * num_step--对列进行反转tf.reverse(inputs, [-1]),
    train_logits, inference_logits = seq2seq_model(inputs,
                                                   targets,
                                                   batch_size,
                                                   source_sequence_len,
                                                   target_sequence_len,
                                                   max_target_sequence_len,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoder_embedding_size,
                                                   decoder_embedding_size,
                                                   rnn_size,
                                                   rnn_num_layers,
                                                   target_vocab_to_int)
    # rnn_output 对应的是32×num_step×vocab_size的   sample_id对应32×num_step 就是最大概率的词
    training_logits = tf.identity(train_logits.rnn_output, name="logits") # 增加一个新节点到gragh中

    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")
    # When using weights as masking, set all valid timesteps to 1 and all padded timesteps to 0,返回的是(target_sequence_len的shape，50)
    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")


    with tf.name_scope("optimization"):# 直接计算序列的损失函数, masks滤去padding的loss计算
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)
# tf.clip_by_value(grad, -1., 1.) 输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。


with tf.Session(graph=train_graph) as sess:
    writer = tf.summary.FileWriter(os.path.join("tmp","logs"), sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths, max_target_len) in enumerate(
                get_batches(source_text_to_int, target_text_to_int, batch_size)):
            # cost是训练decoder时，与target的差
            _, loss = sess.run(
                [train_op, cost],
                {inputs: source_batch,
                 targets: target_batch,
                 learning_rate: lr,
                 source_sequence_len: sources_lengths,
                 target_sequence_len: targets_lengths,
                 max_target_sequence_len : max_target_len})

            if batch_i % display_step == 0 and batch_i > 0:
                # inference_logits 这个地方应该是与验证集的数据去预测，这里用的训练集的，然而好像也没有处理
                batch_train_logits = sess.run(
                    inference_logits,
                    {inputs: source_batch,
                     source_sequence_len: sources_lengths,
                     target_sequence_len: targets_lengths})
                # print('Training:\n')

                # train_source = []
                # for sentence in source_batch:
                #     translate = []
                #     for i in sentence:
                #         if i == 0:
                #             break
                #         translate.append(source_int_to_vocab[i])
                #     train_source.append(translate)
                # for i in train_source:
                #     print(i)
                # print('Targeting:\n')
                tran_all_ = []
                for sentence in target_batch:
                    translate = []
                    for i in sentence:
                        if i == 0:
                            break
                        translate.append(target_int_to_vocab[i])
                    tran_all_.append(translate)

                predict = []
                for sentence in batch_train_logits:
                    translate_pre = []
                    for i in sentence:
                        if i == 0:
                            break
                        translate_pre.append(target_int_to_vocab[i])
                    predict.append(translate_pre)

                # for i in tran_all_:
                #     print(i)
                # print('Testing:\n')
                # tran_all = []
                # for sentence in batch_train_logits:
                #     translate = []
                #     for i in sentence:
                #         if i == 0:
                #             break
                #         translate.append(target_int_to_vocab[i])
                #     tran_all.append(translate)
                # for i in tran_all:
                #     print(i)
                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))
                score = 0
                for (i, j) in list(zip(predict, tran_all_)):
                    score += sentence_bleu(i, j)

                print('Blue is :', score / 256)
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join("tmp","checkpoints","model_fa.ckpt"))
    print('Model Trained and Save to {}'.format(os.path.join("tmp","checkpoints","model_fa.ckpt")))


