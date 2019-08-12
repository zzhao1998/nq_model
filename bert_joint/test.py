# coding=utf-8

import tensorflow as tf
def compute_position_extra_loss(logits,one_hot_positions):
    # 归一化处理
    #logits = tf.nn.log_softmax(logits)


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
    margin = 0.4
    
    # 8.12 update: 这里需要考虑没有答案的情况，这时候positive_logits应该为全0 这时候不计算positive_loss
    # 同时这里需要考虑另一个问题 就是如果positive_logits对应的每一个值都是负数怎么办

    #判断是否有答案吗，如果无答案，不考虑positive loss
    has_answers = tf.reduce_max(onehot_positions[:,1:],axis = -1)  #[batch_size]每一个表示这个是否有答案
    
    positive_inf = 100
    negative_inf = -100
    ones = tf.ones([batch_size,seq_length-1],dtype = tf.float32)
    negative_ones = ones - onehot_positions[:,1:]
    positive_ones = onehot_positions[:,1:]

    behind_logits = logits[:,1:] #对应的是除去CLS的logits
    positive_logits = behind_logits * positive_logits + positive_inf * negative_logits
    negative_logits = behind_logits * negative_logits + negative_inf * positive_logits
    #这里还需要干另外一件事情
    #对于positive logits 其中只有onehot_positions对应的值是非0的，而其他是全为0的，为了避免0项对我们产生影响，我们最好给他们设一个极大的值
    #同理对于negative logits 我们应该正例占有的位置设置一个极小的值

    #positive
    # CLS_P + margin < positive_logits 找到最小的positive logits
    positive_H = tf.reduce_max(margin+CLS_p - positive_logits,axis = -1) # 找到小于CLS+margin 最多处的positive_logits 
    positive_L = tf.nn.relu(positive_H)
    positive_loss =  tf.reduce_mean(positive_L) 
    #negative
    # CLS_P - margin > negative_logits
    negative_H = tf.reduce_max(negative_logits+margin-CLS_p,axis = -1)
    negative_L = tf.nn.relu(negative_H)
    negative_loss =  tf.reduce_mean(negative_L) 
    
    
    return positive_loss+negative_loss


#a = tf.Variable([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
#index_a = tf.Variable([0,2])
 
#b = tf.Variable([1,2,3,4,5,6,7,8,9,10])
#index_b = tf.Variable([2,4,6,8])

logits = tf.Variable([[1.4,2.0,1.0,-0.9],[3.2,4.6,2.3,1.7]],dtype=tf.float32)
one_hot_positions = tf.Variable([[0,0,1,0],[0,0,0,0]],dtype=tf.float32)
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
margin = 0.4

# 8.12 update: 这里需要考虑没有答案的情况，这时候positive_logits应该为全0 这时候不计算positive_loss
# 同时这里需要考虑另一个问题 就是如果positive_logits对应的每一个值都是负数怎么办

#判断是否有答案吗，如果无答案，不考虑positive loss
has_answers = tf.reduce_max(onehot_positions[:,1:],axis = -1)  #[batch_size]每一个表示这个是否有答案

positive_inf = 100
negative_inf = -100
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
positive_loss =  tf.reduce_sum(positive_L*has_answers) /tf.reduce_sum(has_answers)  #这里的特殊之处是不一定每一个都要计算，所以是has_answers
#negative
# CLS_P - margin > negative_logits
negative_H = tf.reduce_max(negative_logits+margin-CLS_p,axis = -1)
negative_L = tf.nn.relu(negative_H)
negative_loss =  tf.reduce_mean(negative_L) 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(has_answers))
    print(sess.run(positive_ones))
    print(sess.run(negative_ones))
    print(sess.run(behind_logits))

    print(sess.run(positive_logits))
    print(sess.run(negative_logits))

    print(sess.run(positive_H))
    print(sess.run(negative_H))

    print(sess.run(positive_L))
    print(sess.run(negative_L))

    print(sess.run(positive_loss))
    print(sess.run(negative_loss))
    #print(sess.run(tf.gather(a, index_a)))
    #print(sess.run(tf.gather(b, index_b)))