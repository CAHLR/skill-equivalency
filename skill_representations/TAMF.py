import json
import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from scipy.sparse import csc_matrix


def l2_normalization(vectors):
    return vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(vectors.shape[0], 1)


def count_skill_and_context(sequences, window_size):
    sc_count = Counter()  
    s_count = Counter()
    c_count = Counter()
    for sequence in sequences:
        for i in range(len(sequence)):
            skill = sequence[i] 
            for j in range(-window_size, window_size+1):
                if i + j >= 0 and i + j < len(sequence) and j != 0:
                    context = sequence[i+j]
                    sc_count[(skill, context)] += 1
                    s_count[skill] += 1
                    c_count[context] += 1
    return sc_count, s_count, c_count


def build_PPMI_matrix(sequences, window_size=5):
    sc_count, s_count, c_count = count_skill_and_context(sequences, window_size)
    data, rows, cols = [], [], []
    total_occurences = sum(sc_count.values())
    for (s, c), n in sc_count.items():
        val = n * total_occurences / (s_count[s] * c_count[c])
        if val > 1:
            data.append(log2(val))
            rows.append(int(s))
            cols.append(int(c))
    PPMI = csc_matrix((data, (rows, cols)))
    return PPMI.toarray()


def get_loss(M, W, H, T, lamb):
    loss_dist = ((M - W.T.dot(H).dot(T)) ** 2).sum()
    loss_reg = (W ** 2).sum() + (H ** 2).sum()
    return loss_dist + (lamb / 2) * loss_reg


def optimizeW(M, H, T, lamb):
    # 2 * H * T * T.T * H.T * W + lamb * W = 2 * H * T * M.T
    a = 2 * H.dot(T).dot(T.T).dot(H.T) + lamb * np.eye(H.shape[0])
    b = 2 * H.dot(T).dot(M.T)
    return np.linalg.solve(a, b)


def optimizeH(M, W, T, lamb):
    # 2 * W * W.T * H * T * T.T + lamb * H = 2 * W * M * T.T
    A = 2 / lamb * W.dot(W.T)
    B = T.dot(T.T)
    C = 2 / lamb * W.dot(M).dot(T.T)
    tensor = np.kron(B.T, A)
    a = tensor + np.eye(tensor.shape[0])
    b = C.reshape(-1, 1, order='F')
    x = np.linalg.solve(a, b)
    return x.reshape(W.shape[0], T.shape[0], order='F')


def optimize_one_turn(M, W, H, T, lamb):
    newW = optimizeW(M, H, T, lamb)
    newH = optimizeH(M, newW, T, lamb)
    return newW, newH


def get_W_and_H(PPMI, content, k, lamb, stop_criterion=1):
    v = PPMI.shape[0]
    f = content.T.shape[0]
    W = np.random.rand(k, v)
    H = np.random.rand(k, f)
    prev_loss = float('inf')
    while True:
        new_loss = get_loss(PPMI, W, H, content.T, lamb)
        print('loss:', new_loss)
        if prev_loss - new_loss < stop_criterion:
            break
        else:
            prev_loss = new_loss
        W, H = optimize_one_turn(PPMI, W, H, content.T, lamb)
    return W, H


def generate_TAMF_vectors_one_taxonomy(sequences_path, skill2id_path, content2vec_path, output_path, k, lamb):
    '''
    Generate TAMF vectors for one taxonomy
    Arguments:
        sequences_path: path to sequences, list of list of strings, json format
                        the sequences are in ids instead of skills to reduce file size
        skill2id_path: path to json file specifying skill-id mapping
        content2vec_path: path to the content2vec file as input to TAMF
        output_path: path to store generated TAMF vectors, csv format
        k: hyperparameter - half vector dimension
        lamb: hyperparameter - regularization coefficient
    Return:
        None, generated vectors are saved to specified output path
    '''
    with open(sequences_path) as f:
        sequences = json.load(f)
    with open(skill2id_path) as f:
        skill2id = json.load(f)
    skills_ordered_by_id = [p[0] for p in sorted(skill2id.items(), key=lambda x: int(x[1]))]
    content2vec = pd.read_csv(content2vec_path, index_col=0)
    content2vec = content2vec.loc[skills_ordered_by_id,:].values
    
    print("Generating PPMI matrix...")
    PPMI = build_PPMI_matrix(sequences)
    skills_ordered_by_id = [p[0] for p in sorted(skill2id.items(), key=lambda x: int(x[1]))]
    content2vec = pd.read_csv(content2vec_path, index_col=0)
    content2vec = content2vec.loc[skills_ordered_by_id,:].values
    
    print("Optimizing matrix factorization...")
    W, H = get_W_and_H(PPMI, content2vec, k, lamb, 10)
    HT = H.dot(content2vec.T)
    TAMF_vec = np.vstack([W, HT]).T
    
    TAMF_vec = pd.DataFrame(TAMF_vec, index=skills_ordered_by_id)
    TAMF_vec.index.name = "skill"
    TAMF_vec = l2_normalization(TAMF_vec)
    TAMF_vec.to_csv(output_path)
    print("Complete!")
    
    
def generate_TAMF_vectors_two_taxonomies(
        src_sequences_path, src_skill2id_path, src_content2vec_path, src_output_path, src_k, src_lamb, 
        dst_sequences_path, dst_skill2id_path, dst_content2vec_path, dst_output_path, dst_k, dst_lamb):
    '''
    Generate TAMF vectors for two taxonomies, trained separately
    Arguments:
        src_sequences_path: path to source sequences, list of list of strings, json format
                            the sequences are in ids instead of skills to reduce file size
        src_skill2id_path: path to json file specifying source skill-id mapping
        src_content2vec_path: path to the source content2vec file as input to TAMF
        src_output_path: path to store generated source TAMF vectors, csv format
        src_k: hyperparameter - source half vector dimension
        src_lamb: hyperparameter - source regularization coefficient
        dst_******: the same set of arguments for destination
    Return:
        None, generated vectors are saved to specified output path
    '''
    print("Generating TAMF vectors for source")
    generate_TAMF_vectors_one_taxonomy(
        src_sequences_path, src_skill2id_path, src_content2vec_path, src_output_path, src_k, src_lamb)
    print("Generating TAMF vectors for destination")
    generate_TAMF_vectors_one_taxonomy(
        dst_sequences_path, dst_skill2id_path, dst_content2vec_path, dst_output_path, dst_k, dst_lamb)
