import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def generate_BOW_vectors_one_taxonomy(problems_path, output_path):
    problems = pd.read_csv(problems_path, sep="\t")
    bow_model = CountVectorizer(ngram_range=(1,1)).fit(problems["content"])
    vectors = pd.DataFrame(bow_model.transform(problems['content']).toarray())
    vectors["skill"] = problems["skill"]
    vectors = vectors.groupby("skill").mean()
    vectors.to_csv(output_path)


def generate_BOW_vectors_two_taxonomies(src_problems_path, dst_problems_path,
                                        src_output_path, dst_output_path):
    src_problems = pd.read_csv(src_problems_path, sep="\t")
    dst_problems = pd.read_csv(dst_problems_path, sep="\t")
    
    combined_problems = src_problems["content"].tolist() + dst_problems["content"].tolist()
    bow_model = CountVectorizer(ngram_range=(1,1)).fit(combined_problems)
    
    src_vectors = pd.DataFrame(bow_model.transform(src_problems['content']).toarray())
    src_vectors["skill"] = src_problems["skill"]
    src_vectors = src_vectors.groupby("skill").mean()
    src_vectors.to_csv(src_output_path)
    
    dst_vectors = pd.DataFrame(bow_model.transform(dst_problems['content']).toarray())
    dst_vectors["skill"] = dst_problems["skill"]
    dst_vectors = dst_vectors.groupby("skill").mean()
    dst_vectors.to_csv(dst_output_path)
