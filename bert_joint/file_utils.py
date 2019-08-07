# coding=utf-8
import os
def read_num_feature_file(file_path):
    f = open(file_path)
    line = f.readline()
    num_list = []
    total_num = 0
    while(line):
        line = line.replace("\n","").replace("  "," ")
        record = line.split(" ")
        record[1] = int(record[1])
        #print(record)
        total_num += record[1]
        line = f.readline()
        if(record[1]==0): #对于是数量为0的文件可以无视掉
            continue
        num_list.append(record)
    #print(num_list)
    #print(total_num)
    f.close()
    return  num_list,total_num

def read_dir(dir_path):
    
    file_list = os.listdir(dir_path)
    #print(file_list)
    if "num_feature" in file_list:
        num_feature_path = os.path.join(dir_path,"num_feature")
        num_list,total_num = read_num_feature_file(num_feature_path)
        for record in num_list:
            #print(record[0])
            if record[0] in file_list:
                record[0] = os.path.join(dir_path,record[0])
            else:
                raise ValueError("%s didn't exist in this folder" %(record[0]))
            

    else:
        raise ValueError("there is no num_feature file in the folder")
    #print(num_list)
    return num_list,total_num


# 这个可以读一个文件，也可以读一个文件夹
# 如果读一个文件 需要指定num_count
# 如果读一个文件夹 需要文件夹中包含num_feature
def read_train_data(train_data_path,num_count):
    if os.path.isdir(train_data_path):
        print("read train data from folder: %s"%(train_data_path))
        return read_dir(train_data_path)
    elif os.path.isfile(train_data_path):
        print("read train data file:%s"%(train_data_path))
        return [train_data_path,num_count],num_count
    else:
        raise ValueError("train_data_path is wrong:%s"%(train_data_path))
"""
train_data_list,total_num = read_train_data("/home/zhangzihao/nq_model/bert_joint/data/no_combination_record/nq-train-00-04.tfrecord",0)
print(train_data_list,total_num)
"""