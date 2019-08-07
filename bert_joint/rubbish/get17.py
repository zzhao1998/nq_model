import jsonlines
count = 0
selection = None
with open("nq-train-sample.jsonl", "r+") as f:
    item = None
    while(True):
        item = f.read()
        count += 1
        if count == 17:
            selection = item
            break
f.close()

with open('output.jsonl',mode='a') as writer:
        writer.write(selection)
writer.close()