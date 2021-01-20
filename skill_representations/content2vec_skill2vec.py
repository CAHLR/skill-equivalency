import numpy as np
import pandas as pd


def l2_normalization(vectors):
    return vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(vectors.shape[0], 1)


def combine_content2vec_and_skill2vec_one_taxonomy(content2vec_path, skill2vec_path, output_path):
    content2vec = pd.read_csv(content2vec_path, index_col=0)
    skill2vec = pd.read_csv(skill2vec_path, index_col=0)
    combined = pd.merge(content2vec, skill2vec, left_index=True, right_index=True, how="outer")
    assert combined.shape[0] == content2vec.shape[0] == skill2vec.shape[0]
    combined.columns = [str(x) for x in range(combined.shape[1])]
    combined = l2_normalization(combined)
    combined.to_csv(output_path)


def combine_content2vec_and_skill2vec_two_taxonomies(
        src_content2vec_path, src_skill2vec_path, src_output_path,
        dst_content2vec_path, dst_skill2vec_path, dst_output_path):
    combine_content2vec_and_skill2vec_one_taxonomy(src_content2vec_path, src_skill2vec_path, src_output_path)
    combine_content2vec_and_skill2vec_one_taxonomy(dst_content2vec_path, dst_skill2vec_path, dst_output_path)
