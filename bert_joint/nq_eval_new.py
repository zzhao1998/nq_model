# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Official evaluation script for Natural Questions.

  https://ai.google.com/research/NaturalQuestions

  ------------------------------------------------------------------------------

  Example usage:

  nq_eval --gold_path=<path-to-gold-files> --predictions_path=<path_to_json>

  This will compute both the official F1 scores as well as recall@precision
  tables for both long and short answers. Note that R@P are only meaningful
  if your model populates the score fields of the prediction JSON format.

  gold_path should point to the five way annotated dev data in the
  original download format (gzipped jsonlines).

  predictions_path should point to a json file containing the predictions in
  the format given below.

  ------------------------------------------------------------------------------

  Prediction format:

  {'predictions': [
    {
      'example_id': -2226525965842375672,
      'long_answer': {
        'start_byte': 62657, 'end_byte': 64776,
        'start_token': 391, 'end_token': 604
      },
      'long_answer_score': 13.5,
      'short_answers': [
        {'start_byte': 64206, 'end_byte': 64280,
         'start_token': 555, 'end_token': 560}, ...],
      'short_answers_score': 26.4,
      'yes_no_answer': 'NONE'
    }, ... ]
  }

  The prediction format mirrors the annotation format in defining each long or
  short answer span both in terms of byte offsets and token offsets. We do not
  expect participants to supply both.

  The order of preference is:

    if start_byte >= 0 and end_byte >=0, use byte offsets,
    else if start_token >= 0 and end_token >= 0, use token offsets,
    else no span is defined (null answer).

  The short answer metric takes both short answer spans, and the yes/no answer
  into account. If the 'short_answers' list contains any non/null spans, then
  'yes_no_answer' should be set to 'NONE'.

  -----------------------------------------------------------------------------

  Metrics:

  If >= 2 of the annotators marked a non-null long answer, then the prediction
  must match any one of the non-null long answers to be considered correct.

  If >= 2 of the annotators marked a non-null set of short answers, or a yes/no
  answer, then the short answers prediction must match any one of the non-null
  sets of short answers *or* the yes/no prediction must match one of the
  non-null yes/no answer labels.

  All span comparisons are exact and each individual prediction can be fully
  correct, or incorrect.

  Each prediction should be provided with a long answer score, and a short
  answers score. At evaluation time, the evaluation script will find a score
  threshold at which F1 is maximized. All predictions with scores below this
  threshold are ignored (assumed to be null). If the score is not provided,
  the evaluation script considers all predictions to be valid. The script
  will also output the maximum recall at precision points of >= 0.5, >= 0.75,
  and >= 0.9.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from interval import Interval, IntervalSet
from collections import OrderedDict
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import eval_utils as util

flags.DEFINE_string(
  'gold_path', 'sample/gold.gz', 'Path to the gzip JSON data. For '
                                 'multiple files, should be a glob '
                                 'pattern (e.g. "/path/to/files-*"')
flags.DEFINE_string('predictions_path', 'sample/pred.json', 'Path to prediction JSON.')
flags.DEFINE_bool(
  'cache_gold_data', False,
  'Whether to cache gold data in Pickle format to speed up '
  'multiple evaluations.')
flags.DEFINE_integer('num_threads', 10, 'Number of threads for reading.')
#flags.DEFINE_bool('pretty_print', True, 'Whether to pretty print output.')
flags.DEFINE_bool('optimal_threshold', True, 'Whether to adjust threshold');

FLAGS = flags.FLAGS


def safe_divide(x, y):
  """Compute x / y, but return 0 if y is zero."""
  if y == 0:
    return 0
  else:
    return x / y

def score_short_answer(gold_label_list, pred_label, threshold = 0):
  """Scores a short answer as correct or not.

  1) First decide if there is a gold short answer with SHORT_NO_NULL_THRESHOLD.
  2) The prediction will get a F1 if:
     a. There is a gold short answer.
     b. The prediction span *set* match exactly with *one* of the non-null gold
        short answer span *set*.

  Args:
    gold_label_list: A list of NQLabel.
    pred_label: A single NQLabel.

  Returns:
    gold_has_answer, pred_has_answer, f1, score
  """

  # There is a gold short answer if gold_label_list not empty and non null
  # answers is over the threshold (sum over annotators).
  gold_has_answer = util.gold_has_short_answer(gold_label_list)

  # There is a pred long answer if pred_label is not empty and short answer
  # set is not empty.

  pred_has_answer = pred_label and (
          (not util.is_null_span_list(pred_label.short_answer_span_list, pred_label.short_score_list, threshold)) or
          pred_label.yes_no_answer != 'none')

  f1 = 0
  p = 0
  r = 0
  # score = pred_label.short_score

  # Both sides have short answers, which contains yes/no questions.
  if gold_has_answer and pred_has_answer:
    if pred_label.yes_no_answer != 'none':  # System thinks its y/n questions.
      for gold_label in gold_label_list:
        if pred_label.yes_no_answer == gold_label.yes_no_answer:
          f1 = 1
          p = 1
          r = 1
          break
    else:
      for gold_label in gold_label_list:
        gold_set = []
        pred_set = []
        for span, score in zip(pred_label.short_answer_span_list, pred_label.short_score_list):
          if score >= threshold:
            pred_set += [(span.start_token_idx, span.end_token_idx)]
        for span in gold_label.short_answer_span_list:
          gold_set += [(span.start_token_idx, span.end_token_idx)]

        def count_same(span_list, interval_set):
          sum = 0
          for span in span_list:
            for interval in interval_set:
              if span[0] == interval[0] and span[1] == interval[1]:
                sum += 1
                break
          return sum
        correct_interval = count_same(pred_set, gold_set)
        precision = safe_divide(correct_interval, len(pred_set))
        recall = safe_divide(correct_interval, len(gold_set))

        if safe_divide(2 * precision * recall, precision + recall) > f1:
          f1 = safe_divide(2 * precision * recall, precision + recall)
          p = precision
          r = recall
  elif not gold_has_answer and not pred_has_answer:
    f1 = 1
    p = 1
    r = 1

  return gold_has_answer, pred_has_answer, f1, p, r

def get_f1(gold_annotation_dict, pred_dict):
  gold_id_set = set(gold_annotation_dict.keys())
  pred_id_set = set(pred_dict.keys())

  if gold_id_set.symmetric_difference(pred_id_set):
    print("gold_id_set", gold_id_set)
    print("pred_id_set", pred_id_set)
    raise ValueError('ERROR: the example ids in gold annotations and example '
                     'ids in the prediction are not equal.')

  final_f1 = 0
  final_p = 0
  final_r = 0
  if FLAGS.optimal_threshold == True:
    score_list = []
    for example_id in gold_id_set:
      gold = gold_annotation_dict[example_id]
      pred = pred_dict[example_id]
      for score in pred.short_score_list:
        score_list += [score]

    score_list += [0]
    score_list.sort()
    #print('score_list', score_list)
    for threshold in score_list:
      sum_f1 = 0
      sum_p = 0
      sum_r = 0
      for example_id in gold_id_set:
        gold = gold_annotation_dict[example_id]
        pred = pred_dict[example_id]
        gold_has_answer, pred_has_answer, f1, p, r = score_short_answer(gold, pred, threshold)
        # print('threshold, f1, gold_has_answer, pred_has_answer', \
        #   threshold, f1, gold_has_answer, pred_has_answer)
        sum_f1 += f1
        sum_p += p
        sum_r += r
      # print('threshold, f1, p, r', \
      #   threshold, safe_divide(sum_f1, len(gold_id_set)), safe_divide(sum_p, len(gold_id_set)), safe_divide(sum_r, len(gold_id_set)))
      if safe_divide(sum_f1, len(gold_id_set)) > final_f1:
        final_f1 = safe_divide(sum_f1, len(gold_id_set))
        final_p = safe_divide(sum_p, len(gold_id_set))
        final_r = safe_divide(sum_r, len(gold_id_set))
  else:
    sum_f1 = 0
    sum_p = 0
    sum_r = 0
    for example_id in gold_id_set:
      gold = gold_annotation_dict[example_id]
      pred = pred_dict[example_id]
      gold_has_answer, pred_has_answer, f1, p, r = score_short_answer(gold, pred)
      sum_f1 += f1
      sum_p += p
      sum_r += r
    final_f1 = safe_divide(sum_f1, len(gold_id_set))
    final_p = safe_divide(sum_p, len(gold_id_set))
    final_r = safe_divide(sum_r, len(gold_id_set))

  return final_f1, final_p, final_r


def main(_):
  cache_path = os.path.join(os.path.dirname(FLAGS.gold_path), 'cache')
  if FLAGS.cache_gold_data and os.path.exists(cache_path):
    logging.info('Reading from cache: %s', format(cache_path))
    nq_gold_dict = pickle.load(open(cache_path, 'r'))
  else:
    nq_gold_dict = util.read_annotation(
      FLAGS.gold_path, n_threads=FLAGS.num_threads)
    if FLAGS.cache_gold_data:
      logging.info('Caching gold data for next time to: %s', format(cache_path))
      pickle.dump(nq_gold_dict, open(cache_path, 'w'))

  nq_pred_dict = util.read_prediction_json(FLAGS.predictions_path)

  ## input: nq_gold_dict, nq_pred_dict
  ## output: long, short score (with optional optimal threshold)

  print('final f1, final_p, final_r', get_f1(nq_gold_dict, nq_pred_dict))

if __name__ == '__main__':
  # flags.mark_flag_as_required('gold_path')
  # flags.mark_flag_as_required('predictions_path')
  app.run(main)
