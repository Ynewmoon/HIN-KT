import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import math
from scipy import sparse
from Product_Layer import pnn1

# load data
data_folder = 'ASSIST2009'
if data_folder == 'ASSIST2009':
    con_sym = '-'
elif data_folder == 'Ednet':
    con_sym = ';'
elif data_folder == 'STATICS2011':
    con_sym = '$$$'
else:
    print('no such dataset!')
    exit()

saved_model_folder = os.path.join(data_folder, 'HIN_model')  # The folder path to save the HIN model
if not os.path.exists(saved_model_folder):
    os.mkdir(saved_model_folder)

pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_pro_skill_sparse.npz'))
skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_skill_skill_sparse.npz'))
pro_pro_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_pro_pro_sparse.npz'))
pro_stu_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_pro_Stu_sparse.npz'))
stu_stu_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_stu_stu_sparse.npz'))

[pro_num, skill_num] = pro_skill_coo.shape
[pro_num, stu_num] = pro_stu_coo.shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
print('problem number %d, student number %d' % (pro_num, stu_num))
print('pro-skill edge %d, pro-pro edge %d, skill-skill edge %d' % (pro_skill_coo.nnz, pro_pro_coo.nnz, skill_skill_coo.nnz))
print('pro-stu edge %d, stu-stu edge %d' % (pro_stu_coo.nnz, stu_stu_coo.nnz))

pro_skill_dense = pro_skill_coo.toarray()
pro_pro_dense = pro_pro_coo.toarray()
skill_skill_dense = skill_skill_coo.toarray()
pro_stu_dense = pro_stu_coo.toarray()
stu_stu_dense = stu_stu_coo.toarray()

pro_feat = np.load(os.path.join(data_folder, 'hxy_pro_feat.npz'))['pro_feat']
print('problem feature shape', pro_feat.shape)
print(pro_feat[:, 0].min(), pro_feat[:, 0].max())
print(pro_feat[:, 1].min(), pro_feat[:, 1].max())

diff_feat_dim = pro_feat.shape[1] - 1

embed_dim = 32
hidden_dim = 128
keep_prob = 0.5
lr = 0.001
bs = 256
epochs = 50
model_flag = 0

tf_pro = tf.placeholder(tf.int32, [None])
tf_stu = tf.placeholder(tf.int32, [None])
tf_diff_feat = tf.placeholder(tf.float32, [None, diff_feat_dim])
tf_pro_skill_targets = tf.placeholder(tf.float32, [None, skill_num], name='tf_pro_skill')
tf_pro_pro_targets = tf.placeholder(tf.float32, [None, pro_num], name='tf_pro_pro')
tf_skill_skill_targets = tf.placeholder(tf.float32, [skill_num, skill_num], name='tf_skill_skill')
tf_pro_stu_targets = tf.placeholder(tf.float32, [None, stu_num], name='tf_pro_stu')
tf_stu_stu_targets = tf.placeholder(tf.float32, [None, stu_num], name='tf_stu_stu')
tf_auxiliary_targets = tf.placeholder(tf.float32, [None], name='tf_auxilary_targets')
tf_keep_prob = tf.placeholder(tf.float32, [1], name='tf_keep_prob')

with tf.variable_scope('pro_skill_embed',reuse=tf.AUTO_REUSE):
    pro_embedding_matrix = tf.get_variable('pro_embed_matrix', [pro_num, embed_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
    skill_embedding_matrix = tf.get_variable('skill_embed_matrix', [skill_num, embed_dim],
                                             initializer=tf.truncated_normal_initializer(stddev=0.1))

    diff_embedding_matrix = tf.get_variable('diff_embed_matrix', [diff_feat_dim, embed_dim],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

    stu_embedding_matrix = tf.get_variable('pro_stu_embed', [stu_num, embed_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
    print(pro_embedding_matrix)
    print(skill_embedding_matrix)
    print(diff_embedding_matrix)
    print(stu_embedding_matrix)

# with tf.variable_scope('pro_stu_embed',reuse=tf.AUTO_REUSE):
#     stu_embedding_matrix = tf.get_variable('pro_stu_embed',[stu_num, embed_dim],
#                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
print(tf_pro)

pro_embed = tf.nn.embedding_lookup(pro_embedding_matrix, tf_pro)
stu_embed = tf.nn.embedding_lookup(stu_embedding_matrix, tf_stu)
diff_feat_embed = tf.matmul(tf_diff_feat, diff_embedding_matrix)

# Hete Correlation Constraint
pro_stu_logits = tf.reshape(tf.matmul(stu_embed, tf.transpose(stu_embedding_matrix)), [-1])
tf_pro_stu_targets_reshape = tf.reshape(tf_pro_stu_targets, [-1])
cross_entropy_pro_stu = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_pro_stu_targets_reshape,logits=pro_stu_logits))
pro_skill_logits = tf.reshape(tf.matmul(pro_embed, tf.transpose(skill_embedding_matrix)), [-1])
tf_pro_skill_targets_reshape = tf.reshape(tf_pro_skill_targets, [-1])
cross_entropy_pro_skill = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_pro_skill_targets_reshape,logits=pro_skill_logits))
L_HC = cross_entropy_pro_stu + cross_entropy_pro_skill


# Homo Similarity Constraint
stu_stu_logits = tf.reshape(tf.matmul(stu_embed, tf.transpose(stu_embedding_matrix)), [-1])
tf_stu_stu_targets_reshape = tf.reshape(tf_stu_stu_targets, [-1])
cross_entropy_stu_stu = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_stu_stu_targets_reshape,logits=stu_stu_logits))
pro_pro_logits = tf.reshape(tf.matmul(pro_embed, tf.transpose(pro_embedding_matrix)), [-1])
tf_pro_pro_targets_reshape = tf.reshape(tf_pro_pro_targets, [-1])
cross_entropy_pro_pro = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_pro_pro_targets_reshape,logits=pro_pro_logits))
skill_skill_logits = tf.reshape(tf.matmul(skill_embedding_matrix, tf.transpose(skill_embedding_matrix)), [-1])
tf_skill_skill_targets_reshape = tf.reshape(tf_skill_skill_targets, [-1])
cross_entropy_skill_skill = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_skill_skill_targets_reshape,logits=skill_skill_logits))
L_HS = cross_entropy_stu_stu + cross_entropy_skill_skill + cross_entropy_pro_pro

skill_embed = tf.matmul(tf_pro_skill_targets, skill_embedding_matrix) / tf.reduce_sum(tf_pro_skill_targets, axis=1, keep_dims=True)
student_embed = tf.matmul(tf_pro_stu_targets, stu_embedding_matrix) / tf.reduce_sum(tf_pro_stu_targets, axis=1, keep_dims=True)
pro_final_embed, p = pnn1([pro_embed, skill_embed, student_embed, diff_feat_embed], embed_dim, hidden_dim, tf_keep_prob[0])
# pro_final_embed= tf.concat([pro_embed, skill_embed, student_embed, diff_feat_embed],1)
L_diff = tf.reduce_mean(tf.square(p-tf_auxiliary_targets))
loss = L_HC + L_HS + L_diff
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

print('finish building graph')
print('-------------------begin training-------------------------------------')


saver = tf.train.Saver()
train_steps = int(math.ceil(pro_num / float(bs)))
with tf.Session() as sess:
    if model_flag == 0:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, os.path.join(saved_model_folder, 'HIN_%d.ckpt' % model_flag))

    print(model_flag)
    print(epochs)

    for i in range(model_flag, epochs):
        train_loss = 0
        for m in range(train_steps):
            b, e = m * bs, min((m + 1) * bs, pro_num)  #bs=256
            b1,e1 = (m * bs)%3840,min((m + 1) * bs, pro_num)%3840
            if e1 == 0:
                e1 = 3840
            batch_pro = np.arange(b, e).astype(np.int32)
            batch_stu = np.arange(b1, e1).astype(np.int32)
            batch_pro_skill_targets = pro_skill_dense[b:e, :]
            batch_pro_pro_targets = pro_pro_dense[b:e, :]
            batch_pro_stu_targets = pro_stu_dense[b:e, :]
            batch_stu_stu_targets = stu_stu_dense[b1:e1, :]
            batch_diff_feat = pro_feat[b:e, :-1]
            batch_auxiliary_targets = pro_feat[b:e, -1]

            feed_dict = {tf_pro: batch_pro,
                         tf_stu: batch_stu,
                         tf_diff_feat: batch_diff_feat,
                         tf_auxiliary_targets: batch_auxiliary_targets,
                         tf_pro_skill_targets: batch_pro_skill_targets,
                         tf_pro_pro_targets: batch_pro_pro_targets,
                         tf_pro_stu_targets: batch_pro_stu_targets,
                         tf_stu_stu_targets: batch_stu_stu_targets,
                         tf_skill_skill_targets: skill_skill_dense,
                         tf_keep_prob: [keep_prob]}

            _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_

        train_loss /= train_steps
        print("epoch %d, loss %.4f" % (i, train_loss))

        if i + 1 in [50, 100, 200, 500, 1000, 1500, 2000]:
            saver.save(sess, os.path.join(saved_model_folder, 'HIN_%d.ckpt' % (i + 1)))

        print('-------------------finish training-------------------------------------')


        pro_repre, skill_repre, stu_repre= sess.run([pro_embedding_matrix, skill_embedding_matrix, stu_embedding_matrix])
        feed_dict = {tf_pro: np.arange(pro_num).astype(np.int32),
                 tf_stu: np.arange(pro_num).astype(np.int32),
                 tf_diff_feat: pro_feat[:, :-1],
                 tf_auxiliary_targets: pro_feat[:, -1],
                 tf_pro_skill_targets: pro_skill_dense,
                 tf_pro_pro_targets: pro_pro_dense,
                 tf_skill_skill_targets: skill_skill_dense,
                 tf_pro_stu_targets: pro_stu_dense,
                 tf_stu_stu_targets: stu_stu_dense,
                 #keep_prob: 0.5,
                 tf_keep_prob: [1.],
                 #tf_keep_prob: [keep_prob]
                 }
        pro_final_repre = sess.run(pro_final_embed, feed_dict=feed_dict)

        with open(os.path.join(data_folder, 'hxy_skill_id_dict.txt'), 'r') as f:
            skill_id_dict = eval(f.read())
        join_skill_num = len(skill_id_dict)
        print('original skill number %d, joint skill number %d' % (skill_num, join_skill_num))

        skill_repre_new = np.zeros([join_skill_num, skill_repre.shape[1]])
        skill_repre_new[:skill_num, :] = skill_repre
        for s in skill_id_dict.keys():
            if con_sym in str(s):
                tmp_skill_id = skill_id_dict[s]
                tmp_skills = [skill_id_dict[ele] for ele in s.split(con_sym)]
                skill_repre_new[tmp_skill_id, :] = np.mean(skill_repre[tmp_skills], axis=0)

        np.savez(os.path.join(data_folder, 'HIN_embedding_%d.npz' % epochs),
                pro_repre=pro_repre, skill_repre=skill_repre_new, pro_final_repre=pro_final_repre)


    print('-------------------store pretrained pro skill stu embedding-------------------------------------')