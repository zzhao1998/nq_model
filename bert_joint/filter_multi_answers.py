# coding=utf-8
# zzh
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import tensorflow as tf
import jsonlines
import json


def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "r"))
    else:
      return tf.gfile.Open(path, "r")
      
def read_jsonlines(filename):
    datas=[]
    file = _open(filename)
    reader = jsonlines.Reader(file)
    for line in reader:
        datas.append(line)
    reader.close()
    return datas

def write_jsonlines(filename,datas):
    with jsonlines.open(filename,mode ='w') as writer:
        for data in datas:
            writer.write(data)
        writer.close()
# select the example with multi short answers
# only consider the first_annotation
def multi_answers_filter(origin_datas):
    multi_answers_datas = []
    for data in origin_datas:
        short_answers=data["annotations"][0]["short_answers"]
        if(len(short_answers)>1):
            multi_answers_datas.append(data)
            #print(short_answers)
    return multi_answers_datas

def read_dir(dirname):
    filenames =tf.gfile.ListDirectory(dirname)
    print(filenames)
    return filenames

def filter_single_multi_answers_json(input_dir,filename,output_dir):
    print("read "+filename+"...")
    origin_datas = read_jsonlines(input_dir+'/'+filename)
    print("origin data size: %d",len(origin_datas))
    #print("finish read "+filename)
    output_filename = "multi_answers_"+filename
    output_datas= multi_answers_filter(origin_datas)
    print("multi answers datas size : %d",len(output_datas))
    if output_filename.endswith(".gz"):
        output_filename = output_filename[:-3]
    write_jsonlines(filename = output_dir+"/"+output_filename,datas=output_datas)
    print("finish write "+output_filename)
    return output_datas

def filter_all_multi_answers_json(dirname,output_dir):
    filenames = read_dir(dirname)
    #all_datas = []
    for filename in filenames:
        datas = filter_single_multi_answers_json(dirname,filename,output_dir)
        #all_datas.extend(datas)
    # sample size 200
    #print("write sample data")
    #sample_datas = all_datas[0:200]
    #write_jsonlines(filename = output_dir+"/sample200.data.jsonl",datas = sample_datas)
    # all data
    #print("write complete data")
    #write_jsonlines(filename = output_dir+"/all.data.jsonl",datas = all_datas)
    #print("complete data size:")
    #print(len(all_datas))

output_dir="multi_answers_data"
input_dir ="/home/data/dataset/NaturalQuestions/v1.0/train"
filter_all_multi_answers_json(input_dir,output_dir)