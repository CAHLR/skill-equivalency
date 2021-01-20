import json
import gensim
import numpy as np
import pandas as pd


def l2_normalization(vectors):
    return vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(vectors.shape[0], 1)


def generate_skill2vec_vectors_one_taxonomy(sequences_path, skill2id_path, output_path, size, window):
    '''
    Generate skill2vec vectors for one taxonomy
    Arguments:
        sequences_path: path to sequences, list of list of strings, json format
                        the sequences are in ids instead of skills to reduce file size
        skill2id_path: path to json file specifying skill-id mapping
        output_path: path to store generated skill2vec vectors, csv format
        size: hyperparameter - vector dimension
        window: hyperparameter - window size
    Return:
        None, generated vectors are saved to specified output path
    '''
    with open(skill2id_path) as f:
        skill2id = json.load(f)
    assert len(set(skill2id.values())) == len(set(skill2id.keys()))
    id2skill = {v: k for k, v in skill2id.items()}
    
    with open(sequences_path) as f:
        sequences = json.load(f)
        
    model = gensim.models.Word2Vec(sequences, size=size, window=window, min_count=0)
    
    ids = sorted(skill2id.values(), key=lambda x: int(x))
    vectors = pd.DataFrame(model.wv[ids], index=ids)
    vectors.index = vectors.index.map(lambda x: id2skill[str(x)])
    vectors.index.name = "skill"
    vectors = l2_normalization(vectors)
    
    vectors.to_csv(output_path)


def generate_skill2vec_vectors_two_taxonomies(
        src_sequences_path, src_skill2id_path, src_output_path, src_size, src_window, 
        dst_sequences_path, dst_skill2id_path, dst_output_path, dst_size, dst_window):
    '''
    Generate skill2vec vectors for two taxonomies, trained separately
    Arguments:
        src_sequences_path: path to source sequences, list of list of strings, json format
                            the sequences are in ids instead of skills to reduce file size
        src_skill2id_path: path to json file specifying source skill-id mapping
        src_output_path: path to store generated source skill2vec vectors, csv format
        src_size: hyperparameter - source vector dimension
        src_window: hyperparameter - source window size
        dst_******: the same set of arguments for destination
    Return:
        None, generated vectors are saved to specified output path
    '''
    generate_skill2vec_vectors_one_taxonomy(
        src_sequences_path, src_skill2id_path, src_output_path, src_size, src_window)
    generate_skill2vec_vectors_one_taxonomy(
        dst_sequences_path, dst_skill2id_path, dst_output_path, dst_size, dst_window)
