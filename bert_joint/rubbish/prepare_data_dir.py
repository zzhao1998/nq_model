# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import random
import run_nq
import tensorflow as tf



flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir", None,
    "Dir which contains Gzipped files containing NQ examples in Json format, one per line.")

flags.DEFINE_string("output_tfrecord", None,
                    "Output tf record file with all features extracted.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels, but include wordpiece to html token maps.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")





def get_examples(input_jsonl_pattern):
    for input_path in tf.gfile.Glob(input_jsonl_pattern):
        with gzip.GzipFile(fileobj=tf.gfile.Open(input_path)) as input_file:
            for line in input_file:
                yield run_nq.create_example_from_jsonl(line)


def convert_jsonl(input_jsonl,creator_fn,instances):
    num_examples = 0
    num_examples_with_correct_context = 0
    print(input_jsonl)
    examples = get_examples(input_jsonl)
    for example in examples:
        num_examples += 1
        for instance in creator_fn.process(example):
            instances.append(instance)
        if num_examples%100==0:
            print("finished_examples: ",num_examples)
        if example["has_correct_context"]:
            num_examples_with_correct_context += 1
        if FLAGS.max_examples > 0 and num_examples >= FLAGS.max_examples:
            break
    tf.logging.info("Examples with correct context retained: %d of %d",num_examples_with_correct_context, num_examples)
    return [num_examples,num_examples_with_correct_context]

def main(_):
    instances = []
    num_examples = 0
    num_examples_with_correct_context = 0
    creator_fn = run_nq.CreateTFExampleFn(is_training=FLAGS.is_training)
    print("start prepare data\n")
    filePath = FLAGS.input_dir
    input_files=os.listdir(filePath)
    for input_file in input_files:
        print(input_file)
        ret = convert_jsonl(FLAGS.input_dir+input_file,creator_fn,instances)
        num_examples+=ret[0]
        num_examples_with_correct_context+=ret[1]
        print("num_examples: ",num_examples)
    print("start writing")
    print(len(instances))
    random.shuffle(instances)
    with tf.python_io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
        for instance in instances:
            writer.write(instance)
        writer.close()
    print("finish writing")
    print("num_examples: ",num_examples)
    print("num_examples_with_correct_context: ",num_examples_with_correct_context)



if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
