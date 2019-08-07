# coding=utf-8
# v1.0 2019.8.5  20:56
import tensorflow as tf

from bert import modeling

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                  mask_positions,use_one_hot_embeddings,mode):
    #print(mode)
    if (mode == "basic"):
        return create_basic_model(bert_config, is_training, input_ids, input_mask, segment_ids,use_one_hot_embeddings)
    if (mode == "mask"):
        return create_mask_model(bert_config, is_training, input_ids, input_mask, segment_ids,mask_positions,use_one_hot_embeddings)

    return create_basic_model(bert_config, is_training, input_ids, input_mask, segment_ids,use_one_hot_embeddings)

def create_mask_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                  mask_positions,use_one_hot_embeddings):
  """Creates a classification model."""

  #print("create mask model ----------------------------------------------")
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/nq/output_weights", [2, hidden_size + 12],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  mask_positions_matrix = tf.cast(tf.reshape(mask_positions, [batch_size * seq_length, 1]),dtype = tf.float32)
  padding = tf.zeros([batch_size * seq_length, 11], dtype=tf.float32)
  mask_positions_matrix = tf.concat([mask_positions_matrix, padding], axis=-1)
  final_hidden_matrix = tf.concat([final_hidden_matrix, mask_positions_matrix], axis=-1)
  final_hidden_matrix = tf.reshape(final_hidden_matrix, [batch_size, seq_length, hidden_size + 12])
  attention_mask = modeling.create_attention_mask_from_input_mask(
            input_ids, input_mask)
  config = bert_config
  all_encoder_layers = modeling.transformer_model(
            input_tensor=final_hidden_matrix,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size + 12, # input hidden size
            num_hidden_layers=1,             #config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)
  #print(all_encoder_layers.shape)
  transformer_output_matrix = all_encoder_layers[-1]

  transformer_output_matrix = tf.reshape(transformer_output_matrix,[batch_size * seq_length, hidden_size+12])
  logits = tf.matmul(transformer_output_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # Get the logits for the answer type prediction.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  return (start_logits, end_logits, answer_type_logits)


'''
bert_config, is_training, input_ids, input_mask, segment_ids,use_one_hot_embeddings 对应的是bert的输入
mask_position 就是掩码向量
'''

def create_basic_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""

  #print("create basic model ----------------------------------------------")
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/nq/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,[batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # Get the logits for the answer type prediction.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  return (start_logits, end_logits, answer_type_logits)


# def create_mask_model(bert_config, is_training, input_ids, input_mask, segment_ids,
#                   mask_positions,use_one_hot_embeddings):
#   """Creates a classification model."""
#   # bert
#   model_bert = modeling.BertModel(
#       config=bert_config,
#       is_training=is_training,
#       input_ids=input_ids,
#       input_mask=input_mask,
#       token_type_ids=segment_ids,
#       use_one_hot_embeddings=use_one_hot_embeddings)
#   # transformer

#   # Get the logits for the start and end predictions.
#   # update : 这里增加了一个mask_position的输入，用来选择已经过滤掉的mask_position
#   final_hidden = model_bert.get_sequence_output()

#   final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
#   batch_size = final_hidden_shape[0]
#   seq_length = final_hidden_shape[1]
#   hidden_size = final_hidden_shape[2]
#   # 这里是预测start_logit和end_logit
#   # 这里要在hidden matrix上额外增加一维mask
#   # 
#   float_mask_positions=tf.cast(mask_positions,dtype=tf.float32)
#   reshape_mask_positions = tf.reshape(float_mask_positions,[batch_size*seq_length,1])

#   output_weights = tf.get_variable(
#       "cls/nq/output_weights", [2, hidden_size+1],
#       initializer=tf.truncated_normal_initializer(stddev=0.02))

#   output_bias = tf.get_variable(
#       "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

#   final_hidden_matrix = tf.reshape(final_hidden,[batch_size * seq_length, hidden_size])
#   final_matrix = tf.concat([final_hidden_matrix,reshape_mask_positions],1)
#   print(final_matrix.shape)

#   logits = tf.matmul(final_matrix, output_weights, transpose_b=True)  
#   logits = tf.nn.bias_add(logits, output_bias)
#   # logits shape :[batch_Size*seq_length,2]
#   logits = tf.reshape(logits, [batch_size, seq_length, 2])
#   logits = tf.transpose(logits, [2, 0, 1])
#   # logits shape :[2,batch_Size,seq_length]
#   unstacked_logits = tf.unstack(logits, axis=0)

#   (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

#   # Get the logits for the answer type prediction.
  
#   answer_type_output_layer = model_bert.get_pooled_output()
#   answer_type_hidden_size = answer_type_output_layer.shape[-1].value

#   num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
#   answer_type_output_weights = tf.get_variable(
#       "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
#       initializer=tf.truncated_normal_initializer(stddev=0.02))

#   answer_type_output_bias = tf.get_variable(
#       "answer_type_output_bias", [num_answer_types],
#       initializer=tf.zeros_initializer())

#   answer_type_logits = tf.matmul(
#       answer_type_output_layer, answer_type_output_weights, transpose_b=True)
#   answer_type_logits = tf.nn.bias_add(answer_type_logits,
#                                       answer_type_output_bias)

#   return (start_logits, end_logits, answer_type_logits)

