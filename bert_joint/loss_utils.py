# coding=utf-8


# loss function
import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("marigin",0.4,"CLS_p 和正确答案和错误答案的距离")

answer_type_len = 5
# 标准的start/end position loss的计算方式
# logits  :  预测得到的的start和end 
# one_hot_positions : 实际的start和end  onehot结构
def compute_loss(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions,mode):
    if(mode == "basic"):
        print("loss mode ------------------------------ basic")
        return compute_loss_basic(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
    if(mode == "hinge"):
        print("loss mode ------------------------------ hinge")
        return compute_loss_hinge(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
    if(mode == "hinge_complete"):
        print("loss mode ------------------------------ hinge_complete")
        return compute_loss_hinge_complete(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
    if(mode == "max_no_answer"):
        print("loss mode ------------------------------ max_no_answer")
        return compute_max_no_answer(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)
    if(mode=="max_no_answer_complete"):
        print("loss mode ------------------------------ max_no_answer_complete")
        return compute_max_no_answer_complete(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions)

def compute_loss_basic(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    start_loss = compute_position_loss(start_logits, start_positions)
    end_loss = compute_position_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_type_positions)
    total_loss = (start_loss + end_loss + answer_type_loss) / 3.0
    return total_loss



def compute_loss_hinge(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    # basic loss
    start_loss = compute_position_loss(start_logits, start_positions)
    end_loss = compute_position_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_type_positions)
    
    # hinge loss
    extra_start_loss = compute_position_hinge_loss(start_logits,start_positions)
    extra_end_loss = compute_position_hinge_loss(end_logits,end_positions)

    total_loss = (start_loss + end_loss + answer_type_loss + extra_start_loss +extra_end_loss) / 5.0
    return total_loss

def compute_loss_hinge_complete(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    extra_start_loss = compute_position_hinge_loss(start_logits,start_positions)
    extra_end_loss = compute_position_hinge_loss(end_logits,end_positions)

    total_loss = (extra_start_loss +extra_end_loss) / 2.0
    return total_loss


def compute_max_no_answer(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    # basic loss
    start_loss = compute_position_loss(start_logits, start_positions)
    end_loss = compute_position_loss(end_logits, end_positions)
    answer_type_loss = compute_label_loss(answer_type_logits, answer_type_positions)
    
    # max no answer loss
    extra_start_loss = compute_position_max_no_answer_loss(start_logits,start_positions)
    extra_end_loss = compute_position_max_no_answer_loss(end_logits,end_positions)

    total_loss = (start_loss + end_loss + answer_type_loss + extra_start_loss +extra_end_loss) / 5.0
    return total_loss

def compute_max_no_answer_complete(start_logits,end_logits,answer_type_logits,start_positions,end_positions,answer_type_positions):
    
    # max no answer loss
    extra_start_loss = compute_position_max_no_answer_loss(start_logits,start_positions)
    extra_end_loss = compute_position_max_no_answer_loss(end_logits,end_positions)

    total_loss = (extra_start_loss +extra_end_loss) / 2.0
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

def compute_position_hinge_loss(logits,one_hot_positions):
    # 归一化处理
    logits = tf.nn.log_softmax(logits)


    CLS_p = logits[:,0]
    #print(logits.shape)
    batch_size = logits.shape[0]
    seq_length = logits.shape[1]
    #print(CLS_p.shape)
    CLS_p = tf.reshape(CLS_p,[-1,1])
    CLS_p = tf.tile(CLS_p,[1,seq_length-1])#扩展

    onehot_positions = tf.cast(one_hot_positions,dtype=tf.float32)
    #positive_logits = (logits * onehot_positions)[:,1:]
    #negative_logits = (logits[:,1:]-positive_logits)

    # hinge loss
    margin = FLAGS.margin
    
    # 8.12 update: 这里需要考虑没有答案的情况，这时候positive_logits应该为全0 这时候不计算positive_loss
    # 同时这里需要考虑另一个问题 就是如果positive_logits对应的每一个值都是负数怎么办

    #判断是否有答案吗，如果无答案，不考虑positive loss
    has_answers = tf.reduce_max(onehot_positions[:,1:],axis = -1)  #[batch_size]每一个表示这个是否有答案
    
    positive_inf = CLS_P + margin
    negative_inf = CLS_p - margin
    ones = tf.ones([batch_size,seq_length-1],dtype = tf.float32)
    negative_ones = ones - onehot_positions[:,1:]
    positive_ones = onehot_positions[:,1:]

    behind_logits = logits[:,1:] #对应的是除去CLS的logits
    positive_logits = behind_logits * positive_ones + positive_inf * negative_ones
    negative_logits = behind_logits * negative_ones + negative_inf * positive_ones
    #这里还需要干另外一件事情
    #对于positive logits 其中只有onehot_positions对应的值是非0的，而其他是全为0的，为了避免0项对我们产生影响，我们最好给他们设一个极大的值
    #同理对于negative logits 我们应该正例占有的位置设置一个极小的值

    #positive
    # CLS_P + margin < positive_logits 找到最小的positive logits
    positive_H = tf.reduce_max(margin+CLS_p - positive_logits,axis = -1) # 找到小于CLS+margin 最多处的positive_logits 
    positive_L = tf.nn.relu(positive_H)
    positive_loss =  tf.reduce_sum(positive_L*has_answers) /tf.reduce_sum(has_answers)


    #negative
    # CLS_P - margin > negative_logits
    negative_H = tf.reduce_max(negative_logits+margin-CLS_p,axis = -1)
    negative_L = tf.nn.relu(negative_H)
    negative_loss =  tf.reduce_mean(negative_L) 
    
    
    return positive_loss+negative_loss




def compute_position_max_no_answer_loss(logits,one_hot_positions):
    # 这个的目标是让CLS_p尽可能大 但是不越过正确答案
    logits = tf.nn.log_softmax(logits)


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
    margin = 0.4 
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
