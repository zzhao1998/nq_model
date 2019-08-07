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
r"""Converts an NQ dataset file to tf examples.

Notes:
  - this program needs to be run for every NQ training shard.
  - this program only outputs the first n top level contexts for every example,
    where n is set through --max_contexts.
  - the restriction from --max_contexts is such that the annotated context might
    not be present in the output examples. --max_contexts=8 leads to about
    85% of examples containing the correct context. --max_contexts=48 leads to
    about 97% of examples containing the correct context.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import random
import run_nq_new
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_jsonl", None,
    "Gzipped files containing NQ examples in Json format, one per line.")

flags.DEFINE_string("output_tfrecord", None,
                    "Output tf record file with all features extracted.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

def _open(input_path):
  if(".gz" in input_path):
    return gzip.GzipFile(fileobj=tf.gfile.Open(input_path))
  else:
    return open(input_path,'r')
def get_examples(input_jsonl_pattern):
  num_examples = 0
  input_file_list = tf.gfile.Glob(input_jsonl_pattern)
  print("input_parttern")
  print(input_file_list)
  for input_path in input_file_list:
    input_file = _open(input_path)
    for line in input_file:
      yield run_nq_new.create_example_from_jsonl(line)
      num_examples = num_examples + 1

def main(_):
  print("start prepare data\n")
  examples_processed = 0
  num_examples_with_correct_context = 0
  creator_fn = run_nq_new.CreateTFExampleFn(is_training=FLAGS.is_training)

  instances = []
  num_examples = 0
  
  examples = get_examples(FLAGS.input_jsonl)#这个部分应该没有问题
  print("input_file: ")
  print(FLAGS.input_jsonl)
  print("output_file: ")
  print(FLAGS.output_tfrecord)
  for example in examples:
    
    num_examples += 1
    for instance in creator_fn.process(example):
      #if(num_examples == 1):
        #print(example)
        #print(len(example["contexts"]))
      instances.append(instance)
      
    if num_examples%100==0:
      print("finished_examples: ",num_examples)
    if example["has_correct_context"]:
      num_examples_with_correct_context += 1
    if examples_processed % 100 == 0:
      tf.logging.info("Examples processed: %d", examples_processed)
    examples_processed += 1
    if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
      break
  tf.logging.info("Examples with correct context retained: %d of %d",
                  num_examples_with_correct_context, examples_processed)

  random.shuffle(instances)
  count = 0
  with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for instance in instances:
      writer.write(instance)
      #print(instance)
      count+=1
  print("num_features:")
  print(count)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
