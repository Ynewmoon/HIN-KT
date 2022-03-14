import os
import sys
import numpy as np
from scipy import sparse
import time


data_folder = 'ASSIST2009'
pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_pro_skill_sparse.npz'))
pro_stu_coo = sparse.load_npz(os.path.join(data_folder, 'hxy_pro_Stu_sparse.npz'))

[pro_num, skill_num] = pro_skill_coo.toarray().shape
[pro_num, stu_num] = pro_stu_coo.toarray().shape
print('problem number %d, skill number %d, student number %d' % (pro_num, skill_num, stu_num))

pro_skill_csc = pro_skill_coo.tocsc()
pro_skill_csr = pro_skill_coo.tocsr()
pro_stu_csc = pro_stu_coo.tocsc()
pro_stu_csr = pro_stu_coo.tocsr()

def extract_pro_pro_sim():
    pro_pro_adj = []
    for p in range(pro_num):
        tmp_skills = pro_skill_csr.getrow(p).indices
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        zipped = zip([p] * similar_pros.shape[0], similar_pros)
        pro_pro_adj += list(zipped)

    pro_pro_adj = list(set(pro_pro_adj))
    pro_pro_adj = np.array(pro_pro_adj).astype(np.int32)
    data = np.ones(pro_pro_adj.shape[0]).astype(np.float32)
    pro_pro_sparse = sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))
    sparse.save_npz(os.path.join(data_folder, 'hxy_pro_pro_sparse.npz'), pro_pro_sparse)
    print("-----extract pro-pro similarity finish！-----")


def extract_skill_skill_sim():
    skill_skill_adj = []
    for s in range(skill_num):
        tmp_pros = pro_skill_csc.getcol(s).indices
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        zipped = zip([s] * similar_skills.shape[0], similar_skills)
        skill_skill_adj += list(zipped)

    skill_skill_adj = list(set(skill_skill_adj))
    skill_skill_adj = np.array(skill_skill_adj).astype(np.int32)
    data = np.ones(skill_skill_adj.shape[0]).astype(np.float32)
    skill_skill_sparse = sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(skill_num, skill_num))
    sparse.save_npz(os.path.join(data_folder, 'hxy_skill_skill_sparse.npz'), skill_skill_sparse)
    print("-----extract skill_skill similarity finish！-----")


def extract_stu_stu_sim():
    stu_stu_adj = []
    for s in range(stu_num):
        tmp_pros = pro_stu_csc.getcol(s).indices
        similar_stus= pro_stu_csr[tmp_pros, :].indices
        zipped = zip([s] * similar_stus.shape[0], similar_stus)
        stu_stu_adj += list(zipped)

    stu_stu_adj = list(set(stu_stu_adj))
    stu_stu_adj = np.array(stu_stu_adj).astype(np.int32)
    data = np.ones(stu_stu_adj.shape[0]).astype(np.float32)
    stu_stu_sparse = sparse.coo_matrix((data, (stu_stu_adj[:, 0], stu_stu_adj[:, 1])),shape=(stu_num, stu_num))
    sparse.save_npz(os.path.join(data_folder, 'hxy_stu_stu_sparse.npz'), stu_stu_sparse)
    print("-----extract stu_stu similarity finish！-----")


extract_pro_pro_sim()
extract_skill_skill_sim()
extract_stu_stu_sim()