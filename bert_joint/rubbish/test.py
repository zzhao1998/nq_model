# coding=utf-8
import run_nq_new
import tensorflow as tf
import gzip
def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "r"))
    else:
      return tf.gfile.Open(path, "r")

def get_examples(path):
    examples = []
    lines = _open(path)
    for line in lines:
        examples.append(run_nq_new.create_example_from_jsonl(line))
    print(examples)

path = "output.jsonl.gz"
get_examples(path)