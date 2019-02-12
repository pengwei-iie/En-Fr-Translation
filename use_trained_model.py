import tensorflow as tf
import pickle
import os

'''
【Input】
  Word Ids:      [120, 89, 131, 217, 171, 38, 180, 204, 208, 89, 103, 192, 213, 177]
  English Words: ['france', 'is', 'never', 'cold', 'during', 'september', ',', 'and', 'it', 'is', 'snowy', 'in', 'october', '.']

【Prediction】
  Word Ids:      [47, 235, 302, 325, 64, 204, 127, 81, 284, 117, 302, 72, 204, 63, 256, 1]
  French Words: ['la', 'france', 'est', 'jamais', 'froid', 'en', 'septembre', ',', 'et', 'il', 'est', 'neigeux', 'en', 'janvier', '.', '<EOS>']

【Full Sentence】
la france est jamais froid en septembre , et il est neigeux en janvier . <EOS>
'''

# 批次大小与训练时保持一致，不可改变
batch_size = 256

# 加载字典
# 加载字典
with open(os.path.join("preparing_resources","enfa_vocab_to_int.pickle"), 'rb') as f:
    source_vocab_to_int = pickle.load(f)
with open(os.path.join("preparing_resources","fa_vocab_to_int.pickle"), 'rb') as f:
    target_vocab_to_int = pickle.load(f)
source_int_to_vocab = {idx: word for word, idx in source_vocab_to_int.items()}
target_int_to_vocab = {idx: word for word, idx in target_vocab_to_int.items()}
print("The size of English Map is : {}".format(len(source_vocab_to_int)))
print("The size of French Map is : {}".format(len(target_vocab_to_int)))

def sentence_to_seq(sentence, source_vocab_to_int):
    """
    将句子转化为数字编码
    """
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence.lower().split()]

    return word_idx

translate_sentence_text = input("请输入句子：")

translate_sentence = sentence_to_seq(translate_sentence_text, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(os.path.join("tmp","checkpoints","model_fa.ckpt.meta"))
    loader.restore(sess, tf.train.latest_checkpoint(os.path.join("tmp","checkpoints")))
    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size})[0]

print('【Input】')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\n【Prediction】')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in translate_logits]))

print("\n【Full Sentence】")
print(" ".join([target_int_to_vocab[i] for i in translate_logits]))