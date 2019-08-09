# coding=utf-8
# 可以重写这个函数
# 输入 
# start_logits 每一个地方是答案开始的概率
# end_logits 每一个地方是答案结束的概率
# unanswerable_p 不是答案的概率
# max_answer_length 每个答案的最大长度
# 注意，这些没有经过softmax，所以求和就可以算概率
#
# 输出
# list = [[start1,end1],[start2,end2]....] 对应所有的答案
# score 对应得分


# predict mode
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("predict_mode", "basic", "the type of predicting method")

"""
class InputFeatures(object):
  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               inside_answer_spans,
               start_position_onehot,
               end_position_onehot,
               mask_onehot,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.inside_answer_spans = inside_answer_spans
    self.start_position_onehot = start_position_onehot
    self.end_position_onehot = end_position_onehot
    self.mask_onehot = mask_onehot
    self.answer_text = answer_text
    self.answer_type = answer_type
"""
class PredictionRecord(object):
  
  def __init__(self):
    self.span_start = None
    self.span_end =None
    self.answerable_p = None
    self.unanswerable_p =None
    self.score = None
    self.answer_type_logits = None

class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None

def compute_predictions(example):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = 10
  max_answer_length = 30
  answers = []
  total_prediction_record_list = []

  for unique_id, result in example.results.iteritems():
    # 每一个feature
    if unique_id not in example.features:
      raise ValueError("No feature found with unique_id:", unique_id)
    token_map = example.features[unique_id]["token_map"].int64_list.value
    start_logits = result["start_logits"]
    end_logits = result["end_logits"]
    answer_type_logits =  result["answer_type_logits"]

    
    prediction_list=get_prediction(start_logits,end_logits,answer_type_logits,max_answer_length,token_map,mode = FLAGS.predict_mode)
    
    # get best prediction
    def get_score(prediction_record):
      return prediction_record.score
    # get all score > 0
    def filter(prediction_record_list):
      ret = []
      for prediction_record in prediction_record_list:
        if(prediction_record.score>0):
          ret.append(prediction_record)
      return ret
    
    #所有score>0均为合格预测

    qualified_prediction_list = filter(prediction_list)

    #将合格预测添加至总预测表中
    total_prediction_record_list.extend(qualified_prediction_list)
  # get sort 排序
  sorted_prediction_list = sorted(total_prediction_record_list,key = get_score ,reverse=True)
    
  
  def get_short_answers_output(prediction_record_list):
    # 这个用以获取输出结果
    short_answers_output = []
    for i in range(len(prediction_record_list)):
      prediction_record = prediction_record_list[i]
      short_answer_record = {
        "start_token":prediction_record.span_start,
        "end_token":prediction_record.span_end,
        "start_byte":-1,
        "end_byte":-1,
        "answerable_p":prediction_record.answerable_p,
        "unanswerable_p":prediction_record.unanswerable_p,
        "score":prediction_record.score
      }
      short_answers_output.append(short_answer_record)
    return short_answers_output



  summary = ScoreSummary()
  summary.predicted_label = {
      "example_id": example.example_id,
      "long_answer": {
          "start_token": 0,
          "end_token": 0,
          "start_byte": -1,
          "end_byte": -1,
      },
      "long_answer_score": 0,
      "short_answers": get_short_answers_output(sorted_prediction_list),
      "short_answers_score": 0,
      "yes_no_answer": "NONE"
  }

  return summary



def get_prediction(start_logits,end_logits,answer_type_logits,max_answer_length,token_map,mode="basic"):
    if (mode == "basic"):
        return get_prediction_basic(start_logits,end_logits,answer_type_logits,max_answer_length,token_map)
    if (mode == "by_start"):
        return get_prediction_by_start(start_logits,end_logits,answer_type_logits,max_answer_length,token_map)
    #else:
        #greedy
        #return get_prediction_basic(start_logits,end_logits,answer_type_logits,max_answer_length,token_map)


"""
def get_prediction_greedy(start_logits,end_logits,max_answer_length):
    answers= []
    length = len(start_logits)
    assert(len(start_logits)==len(end_logits))
    start = 0
    unanswerable_p = start_logits[0]+end_logits[0]
    while (start<length):
        #print(start)
        for i in range(max_answer_length):
            end = start+i+1
            if(end < length and start_logits[start]+end_logits[end] > unanswerable_p):
                #print("-------------------------------------")
                #print(start)
                #print(end)
                answers.append([start,end])
                start = end-1
                #print(start)
                break
        start += 1
    
    max_start_p = max(start_logits)
    max_end_p = max(end_logits)
    score = max_start_p+max_end_p - unanswerable_p
    return answers,score
"""


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


    
def get_prediction_basic(start_logits,end_logits,answer_type_logits,max_answer_length,token_map):
    n_best_size =10
    start_indexes = get_best_indexes(start_logits, n_best_size)
    end_indexes = get_best_indexes(end_logits, n_best_size)
    # 返回的prediction_record_list
    #print(start_logits)
    #print(end_logits)
    #print(answer_type_logits)
    prediction_record_list=[]
    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        if token_map[start_index] == -1:
          continue
        if token_map[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue




        # 完全照着我这里写
        # 每一个答案的得分
        score = start_logits[start_index]+end_logits[end_index] - start_logits[0] - end_logits[0]

        # construct prediction_record
        prediction_record = PredictionRecord()
        prediction_record.span_start = token_map[start_index]
        prediction_record.span_end = token_map[end_index]+1
        prediction_record.answerable_p = start_logits[start_index]+end_logits[end_index]
        prediction_record.unanswerable_p = start_logits[0]+end_logits[0]
        prediction_record.score = score
        prediction_record.answer_type_logits = answer_type_logits
        #print("add one")
        prediction_record_list.append(prediction_record)

    return prediction_record_list






def get_prediction_by_start(start_logits,end_logits,answer_type_logits,max_answer_length,token_map):
  #这个的核心在于只根据start进行选取
  # 如果start>CLS_p则视为答案，选择max_Answer_length以内的最大的end作为答案的end
  prediction_record_list=[]
  length = len(start_logits)
  CLS_p = start_logits[0]
  for i in range(length):
    start_index = i
    if(token_map[i] == -1):
      continue
    if(start_logits[i]>CLS_p):
      print(token_map[i])
      #进行选择最大的end
      end = -1
      end_p = -10000
      for j in range(max_answer_length):
        if(i+j>=length):
          break
        if(token_map[i+j] == -1):
          continue
        if(end_logits[i+j]>end_p):
          end = i+j
          end_p = end_logits[i+j]
      if(end == -1):
        continue
      else:
        end_index = end


      score = start_logits[start_index]+end_logits[end_index] - start_logits[0] - end_logits[0]
      # construct prediction_record
      prediction_record = PredictionRecord()
      prediction_record.span_start = token_map[start_index]
      prediction_record.span_end = token_map[end_index]+1
      prediction_record.answerable_p = start_logits[start_index]+end_logits[end_index]
      prediction_record.unanswerable_p = start_logits[0]+end_logits[0]
      prediction_record.score = score
      prediction_record.answer_type_logits = answer_type_logits
      prediction_record_list.append(prediction_record)
      
  return prediction_record_list