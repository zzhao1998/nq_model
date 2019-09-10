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

multi_answer_sample_path ="/home/zhangzihao/nq_model/bert_joint/data/multi_answers_sample.jsonl.gz"
# complete orig data path
# 00-44用于train  45-49用于valid
#train_data_path="/home/data/share/NaturalQuestions/v1.0/train/"
valid_data_path=dataset_position+"train/nq-train-4[5-9].jsonl.gz"


# tfrecord
train_no_combination_tfrecord_path = "/home/zhangzihao/nq_model/bert_joint/data/train_tfrecord" #这个是部分数据集00-44
train_combination_tfrecord_path = "/home/zhangzihao/nq_model/bert_joint/data/combination_record"


#model
bert_base_path ="/home/zhangzihao/nq_model/bert_joint/bert_base/bert_model.ckpt"


no_combination_loss_basic_model_path="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_basic_epoch1-2/model.ckpt-128231"
no_combination_loss_advance_model_path ="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_advance_marigin_0.4_epoch1/model.ckpt-128231"
no_combination_loss_cross_model_path ="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_cross/model.ckpt-64116"
bert_model_output_dir = "/home/zhangzihao/nq_model/bert_joint/bert_model_output/"
hinge_cross = bert_model_output_dir+"model_no_combination_loss_hinge_cross_epoch1/model.ckpt-21402"
wrong_model="/home/zhangzihao/nq_model/bert_joint/bert_model_output/model_no_combination_loss_hinge_margin_0.4_epoch1/model.ckpt-35687"

if operation == "predict":
    train_mode = raw_input('please input train_mode: ')
    
    #pretrained
    if(train_mode == "pretrained"):

        predict_file_path = multi_answer_sample_path
        #predict_file_path =test_sample_path

        loss_mode = raw_input('please input loss_mode(basic/max_no_answer/hinge/combination/old): ')
        margin = input('please input margin:')
        predict_mode ='by_start'
        # predict
        output_prediction_file ="prediction/pretrained_multi_answer/{}_margin_{}_epoch1_{}.json".format(loss_mode,margin,predict_mode)

        model_dir = "bert_model_output/model_pretrained_loss_"+loss_mode+"_margin_"+str(margin)+"_epoch1"
        #model_dir = no_combination_loss_basic_model_path
        
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
        command = "python -m nq_eval_new \
        --logtostderr \
        --gold_path={} \
        --predictions_path={} \
        --optimal_threshold=True".format(predict_file_path,output_prediction_file)
        print(command)
        os.system(command)

        # eval
        command = "python -m nq_eval_new \
        --logtostderr \
        --gold_path={} \
        --predictions_path={} \
        --optimal_threshold=False".format(predict_file_path,output_prediction_file)
        print(command)
        os.system(command)

    
    if(train_mode == "origin"):

        predict_file_path = dev_sample_path
        #predict_file_path =test_sample_path

        loss_mode = raw_input('please input loss_mode(basic/max_no_answer/hinge/combination/old): ')
        margin = input('please input margin:')
        predict_mode ='by_start'
        # predict
        output_prediction_file ="prediction/pretrained_dev/origin_{}_margin_{}_epoch1_{}.json".format(loss_mode,margin,predict_mode)

        model_dir = "bert_model_output/model_pretrained_loss_"+loss_mode+"_margin_"+str(margin)+"_epoch1"
        #model_dir = no_combination_loss_basic_model_path
        
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
        
        
        
        # origin nq_eval
        command = "python -m nq_eval \
        --logtostderr \
        --gold_path={} \
        --predictions_path={} ".format(predict_file_path,output_prediction_file)
        print(command)
        os.system(command)

        
    

    #start
    if train_mode == 'start':

        predict_file_path = dev_sample_path
        
        loss_mode = raw_input('please input loss_mode(basic/max_no_answer/hinge): ')
        margin = input('please input margin:')
        predict_mode ='by_start'
        # predict
        output_prediction_file ="prediction/pretrained_dev/start_{}_margin_{}_epoch1_{}.json".format(loss_mode,margin,predict_mode)

        #model_dir = no_combination_loss_basic_model_path
        model_dir = "bert_model_output/model_start_loss_"+loss_mode+"_margin_"+str(margin)+"_epoch1"
        #model_dir = "bert_model_output/model_no_combination_loss_hinge_margin_0.4_epoch1_wrong/model.ckpt-21402"
        
        
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
        command = "python -m nq_eval_new \
        --logtostderr \
        --gold_path={} \
        --predictions_path={} \
        --optimal_threshold=True".format(predict_file_path,output_prediction_file)
        print(command)
        os.system(command)

        # eval
        command = "python -m nq_eval_new \
        --logtostderr \
        --gold_path={} \
        --predictions_path={} \
        --optimal_threshold=False".format(predict_file_path,output_prediction_file)
        print(command)
        os.system(command)

        


if operation == "train":
    train_mode = raw_input('please input train_mode: ')
    train_tfrecord_path = train_no_combination_tfrecord_path
    #init_checkpoint = "bert_model_output/model_no_combination_loss_basic_epoch1/model.ckpt-71089"

    """
    if train_mode == "basic":
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
    if train_mode== "advance":
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


    """
    #用于先预训练 然后再进行特殊loss的训练
    if train_mode== "start":
        
        # 已经训练了一个周期的
        init_checkpoint = bert_base_path
        loss_mode = raw_input('please input loss_mode(basic/max_no_answer/hinge): ')

        margin = input('please input margin:')
        output_dir = "bert_model_output_new/model_start_loss_"+loss_mode+"_margin_"+str(margin)+"_epoch1"
        
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
        --loss_mode={} \
        --margin={}".format(gpu_id,train_tfrecord_path,init_checkpoint,output_dir,loss_mode,margin)


        print(command)
        os.system(command)
    #用于先预训练 然后再进行特殊loss的训练
    if train_mode== "pretrained":
        
        # 已经训练了一个周期的
        init_checkpoint ="/home/zhangzihao/nq_model/bert_joint/bert_model_output_new/model_start_loss_basic_margin_0_epoch1/model.ckpt-64116"
        loss_mode = raw_input('please input loss_mode(basic/max_no_answer/hinge): ')

        margin = input('please input margin:')
        output_dir = "bert_model_output_new/model_pretrained_loss_"+loss_mode+"_margin_"+str(margin)+"_epoch1"
        
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
        --loss_mode={} \
        --margin={}".format(gpu_id,train_tfrecord_path,init_checkpoint,output_dir,loss_mode,margin)


        print(command)
        os.system(command)