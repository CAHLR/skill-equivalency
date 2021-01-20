import gensim
import numpy as np
import pandas as pd


def l2_normalization(vectors):
    return vectors / np.linalg.norm(vectors, ord=2, axis=1).reshape(vectors.shape[0], 1)


def get_problem_vector(content, model, size):
    content = [word for word in content if word in model.wv]
    if len(content) == 0:
        return np.random.rand(size)
    return model.wv[content].mean(axis=0)


def get_skill_vector(group):
    return np.array(group["vector"].tolist()).mean(axis=0)


def generate_content2vec_vectors_one_taxonomy(
        problems_path, output_path,
        size, window, min_count):
    '''
    Generate content2vec vectors for one taxonomy
    Arguments:
        problems_path: path to problems, must contain columns "skill" and "content", tsv format
        output_path: path to store generated content2vec vectors, csv format
        size: hyperparameter - vector dimension
        window: hyperparameter - window size
        min_count: hyperparameter - minimum count
    Return:
        None, generated vectors are saved to specified output path
    '''
    problems = pd.read_csv(problems_path, sep="\t")
    problems["content"] = problems["content"].apply(lambda x: x.split())
    
    corpus = problems["content"].tolist()
    model = gensim.models.Word2Vec(corpus, size=size, window=window, min_count=min_count, iter=50)
    
    problems["vector"] = problems["content"].apply(get_problem_vector, args=(model,size))
    skills = problems.groupby("skill").apply(get_skill_vector)
    skills = pd.DataFrame(skills.tolist(), index=skills.index)
    skills = l2_normalization(skills)
    
    skills.to_csv(output_path)


def generate_content2vec_vectors_two_taxonomies(
        src_problems_path, dst_problems_path,
        src_output_path, dst_output_path,
        size, window, min_count):
    '''
    Generate content2vec vectors for two taxonomies, trained on the joint problem corpus
    Arguments:
        src_problems_path: path to source problems, must contain columns "skill" and "content", tsv format
        dst_problems_path: path to destination problems, must contain columns "skill" and "content", tsv format
        src_output_path: path to store generated source content2vec vectors, csv format
        dst_output_path: path to store generated destination content2vec vectors, csv format
        size: hyperparameter - vector dimension
        window: hyperparameter - window size
        min_count: hyperparameter - minimum count
    Return:
        None, generated vectors are saved to specified output path
    '''
    src_problems = pd.read_csv(src_problems_path, sep="\t")
    src_problems["content"] = src_problems["content"].apply(lambda x: x.split())

    dst_problems = pd.read_csv(dst_problems_path, sep="\t")
    dst_problems["content"] = dst_problems["content"].apply(lambda x: x.split())
    
    corpus = src_problems["content"].tolist() + dst_problems["content"].tolist()
    model = gensim.models.Word2Vec(corpus, size=size, window=window, min_count=min_count, iter=50)
    
    src_problems["vector"] = src_problems["content"].apply(get_problem_vector, args=(model,size))
    src_skills = src_problems.groupby("skill").apply(get_skill_vector)
    src_skills = pd.DataFrame(src_skills.tolist(), index=src_skills.index)
    src_skills = l2_normalization(src_skills)

    dst_problems["vector"] = dst_problems["content"].apply(get_problem_vector, args=(model,size))
    dst_skills = dst_problems.groupby("skill").apply(get_skill_vector)
    dst_skills = pd.DataFrame(dst_skills.tolist(), index=dst_skills.index)
    dst_skills = l2_normalization(dst_skills)
    
    src_skills.to_csv(src_output_path)
    dst_skills.to_csv(dst_output_path)
