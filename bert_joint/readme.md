

# Multi-Spans Prediction based on BERT


## file constructure

文件夹:
- bert_base/ 存放原始的bert模型参数
- bert_model_output/ 存放训练后的模型参数
- data/ 存放 原始数据和生成的TFRecord
- command/ 存放命令
- rubbish/ 回收站
- prediction/ 在预测过程中存放生成的预测结果

无关文件：
- bert_config.json bert模型的结构参数，最好不要改
- _init_.py  不用改

数据生成：
- prepare_nq_data.py: 根据原始数据生成TFrecord

模型训练
- run_nq.py: 原始的baseline模型
- run_nq_multi_answer: 用单任务模型预测多答案
- run_nq_new： 用多任务模型预测多答案

预测
- answer_predict_utils.py: 选择预测方法

评价
eval_utils.py 评价函数
nq_eval.py 原始的评价方法
nq_eval_new 改进的评价方法(支持no answer 支持multi answers)





