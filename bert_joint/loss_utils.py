# coding=utf-8


# loss function
import tensorflow as tf


answer_type_len = 5
# 标准的start/end position loss的计算方式
# logits  :  预测得到的的start和end 
# one_hot_positions : 实际的start和end  onehot结构
def compute_loss(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions,mode):
    if(mode == "basic"):
        return compute_loss_basic(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
    if(mode == "advance"):
        return compute_loss_advance(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)

    return compute_loss_basic(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
def compute_loss_basic(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    start_loss = compute_position_loss(start_logits, start_positions)
    end_loss = compute_position_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_type_positions)
    total_loss = (start_loss + end_loss + answer_type_loss) / 3.0
    return total_loss

def compute_loss_advance(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    start_loss = compute_position_loss(start_logits, start_positions)
    end_loss = compute_position_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_type_positions)
    
    extra_start_loss = compute_position_extra_loss(start_logits,start_positions)
    extra_end_loss = compute_position_extra_loss(end_logits,end_positions)
    total_loss = (start_loss + end_loss + answer_type_loss + extra_start_loss +extra_end_loss) / 5.0
    return total_loss
def compute_position_loss(logits, one_hot_positions):
    positions = tf.cast(one_hot_positions,dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(
        tf.reduce_sum(positions * log_probs, axis=-1))
    return loss


# label loss
# logits  :  预测得到的的answer_type
# one_hot_positions : 实际的start和end  onehot结构
def compute_label_loss(logits, labels):
    one_hot_labels = tf.one_hot(
        labels, depth=answer_type_len, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    loss = -tf.reduce_mean(
        tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
    return loss



#这个要求计算多个值
# logits [batch_size,seq_length]
# onehot_positions [batch_size,seq_length]

def compute_position_extra_loss(logits,one_hot_positions):
    # 不归一化处理
    # logits = tf.nn.log_softmax(logits)


    CLS_p = logits[:,0]
    #print(logits.shape)
    batch_size = logits.shape[0]
    seq_length = logits.shape[1]
    #print(CLS_p.shape)
    CLS_p = tf.reshape(CLS_p,[-1,1])
    CLS_p = tf.tile(CLS_p,[1,seq_length-1])

    onehot_positions = tf.cast(one_hot_positions,dtype=tf.float32)
    positive_logits = (logits * onehot_positions)[:,1:]
    negative_logits = (logits[:,1:]-positive_logits)
    #print(positive_logits.shape) 
    #print(negative_logits.shape)

    # hinge loss
    margin = 0.5
    #positive
    # CLS_P + margin < positive_logits
    positive_H = tf.reduce_max(margin+CLS_p - positive_logits,axis = -1) # 找到小于CLS+margin 最多处的positive_logits 
    positive_L = tf.nn.relu(positive_H)
    positive_loss =  tf.reduce_mean(positive_L) 
    #negative
    # CLS_P - margin > negative_logits
    negative_H = tf.reduce_max(negative_logits+margin-CLS_p,axis = -1)
    negative_L = tf.nn.relu(negative_H)
    negative_loss =  tf.reduce_mean(negative_L) 
    
    
    return positive_loss+negative_loss


