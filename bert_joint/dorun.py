# coding=utf-8
import os


gpu_id = int(input('please input gpu id 0~3: '))
operation = raw_input('please input operation(train/predict/prepare_data): ')


#
dataset_position ="/home/data/share/NaturalQuestions/v1.0/"
# sample orig data path
dev_sample_path =dataset_position+"sample/nq-dev-sample.jsonl.gz"
train_sample_path = dataset_position+"sample/nq-train-sample.jsonl.gz"
test_sample_path = "test/output.jsonl.gz" # (456 457)(470 471) (489 491) 

# complete orig data path
# 00-44用于train  45-49用于valid
#train_data_path="/home/data/share/NaturalQuestions/v1.0/train/"
valid_data_path=dataset_position+"train/nq-train-4[5-9].jsonl.gz"


# tfrecord
train_no_combination_tfrecord_path = "/home/zhangzihao/nq_model/bert_joint/data/no_combination_record_train" #这个是部分数据集00-44
train_combination_tfrecord_path = "/home/zhangzihao/nq_model/bert_joint/data/combination_record"


#model
bert_base_path ="/home/zhangzihao/nq_model/bert_joint/bert_base/bert_model.ckpt"


no_combination_loss_basic_model_path="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_basic_epoch1/model.ckpt-128231"
no_combination_loss_advance_model_path ="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_advance_marigin_0.4_epoch1/model.ckpt-128231"
no_combination_loss_cross_model_path ="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_cross/model.ckpt-64116"
bert_model_output_dir = "/home/zhangzihao/nq_model/bert_joint/bert_model_output/"
hinge_cross = bert_model_output_dir+"model_no_combination_loss_hinge_cross_epoch1/model.ckpt-21402"
wrong_model="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_hinge_margin_0.4_epoch1/model.ckpt-35687"

if operation == "predict":

    predict_file_path = dev_sample_path

    predict_mode ='by_start'
    # predict
    output_prediction_file ="prediction/cross35687-{}.json".format(predict_mode)

    #model_dir = no_combination_loss_basic_model_path
    model_dir = wrong_model
    #model_dir = "bert_model_output/model_no_combination_loss_hinge_margin_0.4_epoch1/model.ckpt-21402"
    
    
    command = "CUDA_VISIBLE_DEVICES={} python2 -m run_nq_new \
    --logtostderr \
    --bert_config_file=bert_config.json \
    --vocab_file=./data/vocab-nq.txt \
    --predict_file={} \
    --init_checkpoint={} \
    --do_predict \
    --output_dir=./fun \
    --model_mode='basic' \
    --loss_mode='advance' \
    --predict_mode={} \
    --num_best=0 \
    --output_prediction_file={}".format(gpu_id,predict_file_path,model_dir,predict_mode,output_prediction_file)
    
    print(command)
    os.system(command)
    
    
    
    # eval
    command = "python -m nq_eval2 \
    --logtostderr \
    --gold_path={} \
    --predictions_path={}".format(predict_file_path,output_prediction_file)
    print(command)
    os.system(command)
    




if operation == "train":
    loss_mode = raw_input('please input loss_mode: ')
    train_tfrecord_path = train_no_combination_tfrecord_path
    #init_checkpoint = "bert_model_output/model_no_combination_loss_basic_epoch1/model.ckpt-71089"
    if loss_mode == "basic":
        init_checkpoint = bert_base_path
        output_dir = "bert_model_output/model_no_combination_loss_basic_epoch1"
        
        command = "CUDA_VISIBLE_DEVICES={} python2 -m run_nq_new \
        --logtostderr \
        --bert_config_file=./bert_config.json \
        --vocab_file=./data/vocab-nq.txt \
        --train_precomputed_file={} \
        --train_num_precomputed=42360 \
        --learning_rate=1e-5 \
        --num_train_epochs=2 \
        --max_seq_length=512 \
        --train_batch_size=6 \
        --save_checkpoints_steps=5000 \
        --init_checkpoint={} \
        --do_train \
        --output_dir={} \
        --model_mode='basic' \
        --loss_mode='basic'".format(gpu_id,train_tfrecord_path,init_checkpoint,output_dir)

        print(command)
        os.system(command)
    if loss_mode== "advance":
        init_checkpoint = bert_base_path
        output_dir = "bert_model_output/model_no_combination_loss_hinge_marigin_0.4_epoch2"
        
        command = "CUDA_VISIBLE_DEVICES={} python2 -m run_nq_new \
        --logtostderr \
        --bert_config_file=./bert_config.json \
        --vocab_file=./data/vocab-nq.txt \
        --train_precomputed_file={} \
        --train_num_precomputed=42360 \
        --learning_rate=1e-5 \
        --num_train_epochs=2 \
        --max_seq_length=512 \
        --train_batch_size=6 \
        --save_checkpoints_steps=5000 \
        --init_checkpoint={} \
        --do_train \
        --output_dir={} \
        --model_mode='basic' \
        --loss_mode='advance' \
        --predict_mode='by_start' \
        --margin=0.4".format(gpu_id,train_tfrecord_path,init_checkpoint,output_dir)

        print(command)
        os.system(command)

    if loss_mode== "cross":
        init_checkpoint = no_combination_loss_basic_model_path
        output_dir = "bert_model_output/model_no_combination_loss_hinge_cross_epoch1"
        
        command = "CUDA_VISIBLE_DEVICES={} python2 -m run_nq_new \
        --logtostderr \
        --bert_config_file=./bert_config.json \
        --vocab_file=./data/vocab-nq.txt \
        --train_precomputed_file={} \
        --train_num_precomputed=42360 \
        --learning_rate=1e-5 \
        --num_train_epochs=1 \
        --max_seq_length=512 \
        --train_batch_size=6 \
        --save_checkpoints_steps=5000 \
        --init_checkpoint={} \
        --do_train \
        --output_dir={} \
        --model_mode='basic' \
        --loss_mode='advance' \
        --margin=0.4".format(gpu_id,train_tfrecord_path,init_checkpoint,output_dir)


        print(command)
        os.system(command)