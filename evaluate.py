import torch
import numpy as np
import pandas as pd


def l2_normalization(vectors):
    return vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(vectors.shape[0], 1)


def get_rankings(src, dst):
    x = torch.tensor(src.to_numpy(), dtype=torch.double)
    y = torch.tensor(dst.to_numpy(), dtype=torch.double)
    new_x = x.unsqueeze(1).repeat(1, y.size(0), 1)
    new_y = y.unsqueeze(0).repeat(x.size(0), 1, 1)
    cos_sim = torch.mul(new_x, new_y).sum(axis=2)
    _, indices = torch.sort(cos_sim, descending=True)
    rankings = {}
    for i in range(len(src)):
        rankings[src.index[i]] = [dst.index[idx] for idx in indices[i]]
    return rankings


def get_recall_at_5(rankings, labels):
    accurate_count = 0
    labels_tuple = [(x[0], x[1]) for x in labels[["source", "destination"]].values]
    for src_skill in rankings:
        for dst_skill in rankings[src_skill][:5]:
            if (src_skill, dst_skill) in labels_tuple:
                accurate_count += 1
    return accurate_count / len(labels)


def get_mean_reciprocal_rank(rankings, labels):
    reciprocal_ranks = []
    labels_tuple = [(x[0], x[1]) for x in labels[["source", "destination"]].values]
    for src_skill in rankings:
        relevant = [x[1] for x in labels_tuple if x[0] == src_skill]
        if len(relevant) != 0:
            rank = np.min([np.where(np.array(rankings[src_skill]) == relevant[k]) for k in range(len(relevant))]) + 1
            reciprocal_ranks.append(1 / rank)
    return np.mean(reciprocal_ranks)


def evaluate(src_vectors_path, dst_vectors_path, labels_path, recall_at_5=True, mrr=True):
    '''
    Arguments:
        src_vectors_path: path to source skill vectors, the first column must be skill names
        dst_vectors_path: path to destination skill vectors, the first column must be skill names
        labels_path: path to ground truth labels for evaluation, must contain columns "source" and "destination"
        recall_at_5: whether to output recall@5
        mrr: whether to output mean reciprocal rank
    Return:
        dict with specified results
    '''
    src_vectors = pd.read_csv(src_vectors_path, index_col=0)
    src_vectors = l2_normalization(src_vectors)
    dst_vectors = pd.read_csv(dst_vectors_path, index_col=0)
    dst_vectors = l2_normalization(dst_vectors)
    rankings = get_rankings(src_vectors, dst_vectors)
    labels = pd.read_csv(labels_path)
    results = {}
    if recall_at_5:
        results["recall_at_5"] = get_recall_at_5(rankings, labels)
    if mrr:
        results["mrr"] = get_mean_reciprocal_rank(rankings, labels)
    return results
