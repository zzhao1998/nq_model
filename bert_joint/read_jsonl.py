# coding=utf-8
import json
import gzip

filename = "test/output.jsonl.gz"
with open(filename) as fileobj:
    jsonl_list = None
    if ".gz" in filename:
        jsonl_list = gzip.GzipFile(fileobj=fileobj)
    else:
        jsonl_list = fileobj
    for line in jsonl_list:
        json_example = json.loads(line)
        tokens = json_example["document_tokens"]
        tokens_list= []
        for token in tokens:
            tokens_list.append(token["token"])
        while(True):
            start = int(input("start:"))
            end = int(input("end"))
            content = " ".join(tokens_list[start:end])
            print(content)