import os
import torch
import argparse


def parser():
    parser = argparse.ArgumentParser()
    
    # Experiment settings
    models = ["BOW", "TFIDF", "content2vec", "skill2vec", "content2vec_skill2vec", "TAMF"]
    parser.add_argument("--src-name", type=str, default="source", help="source platform name")
    parser.add_argument("--dst-name", type=str, default="destination", help="destination platform name")
    parser.add_argument("--src-repr-model", type=str, choices=models, required=True, help="source representation model")
    parser.add_argument("--dst-repr-model", type=str, choices=models, required=True, help="destination representation model")
    parser.add_argument("--expr-name", type=str, help="a custom experiment name")
    
    # Input data
    parser.add_argument("--labels-path", type=str, required=True, help="path to ground truth labels")
    parser.add_argument("--src-problems-path", type=str, help="path to source problems")
    parser.add_argument("--dst-problems-path", type=str, help="path to destination problems")
    parser.add_argument("--src-sequences-path", type=str, help="path to source sequences")
    parser.add_argument("--dst-sequences-path", type=str, help="path to destination sequences")
    parser.add_argument("--src-skill2id-path", type=str, help="path to source skill2id file")
    parser.add_argument("--dst-skill2id-path", type=str, help="path to destination skill2id file")
    
    # Output root dir
    parser.add_argument("--output-root", type=str, default="./output", help="path to output root directory")
    
    # Hyperparameters
    parser.add_argument("--content2vec-size", type=int, default=50, help="content2vec vector size, only used when both --src-repr-model and --dst-repr-model are content2vec")
    parser.add_argument("--content2vec-window", type=int, default=20, help="content2vec window size, only used when both --src-repr-model and --dst-repr-model are content2vec")
    parser.add_argument("--content2vec-min-count", type=int, default=30, help="content2vec minimum count, only used when both --src-repr-model and --dst-repr-model are content2vec")
    
    parser.add_argument("--src-content2vec-size", type=int, default=50, help="source content2vec vector size, used when --src-repr-model is content2vec and --dst-repr-model is not content2vec")
    parser.add_argument("--src-content2vec-window", type=int, default=20, help="source content2vec window size, used when --src-repr-model is content2vec and --dst-repr-model is not content2vec")
    parser.add_argument("--src-content2vec-min-count", type=int, default=30, help="source content2vec minimum count, used when --src-repr-model is content2vec and --dst-repr-model is not content2vec")
    parser.add_argument("--dst-content2vec-size", type=int, default=50, help="destination content2vec vector size, used when --dst-repr-model is content2vec and --src-repr-model is not content2vec")
    parser.add_argument("--dst-content2vec-window", type=int, default=20, help="destination content2vec window size, used when --dst-repr-model is content2vec and --src-repr-model is not content2vec")
    parser.add_argument("--dst-content2vec-min-count", type=int, default=30, help="destination content2vec minimum count, used when --dst-repr-model is content2vec and --src-repr-model is not content2vec")
    
    parser.add_argument("--src-skill2vec-size", type=int, default=50, help="source skill2vec vector size")
    parser.add_argument("--src-skill2vec-window", type=int, default=20, help="source skill2vec window size")
    parser.add_argument("--dst-skill2vec-size", type=int, default=50, help="destination skill2vec vector size")
    parser.add_argument("--dst-skill2vec-window", type=int, default=20, help="destination skill2vec window size")
    
    parser.add_argument("--src-TAMF-k", type=int, default=100, help="source TAMF half vector size k")
    parser.add_argument("--src-TAMF-lambda", type=float, default=0.1, help="source TAMF regularization coefficient lambda")
    parser.add_argument("--dst-TAMF-k", type=int, default=100, help="destination TAMF half vector size k")
    parser.add_argument("--dst-TAMF-lambda", type=float, default=0.1, help="destination TAMF regularization coefficient lambda")
    
    # Translation parameters
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="device to run translation on")
    parser.add_argument("--num-epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="percentage of validation set")
    parser.add_argument("--max-val-non-decreasing-epochs", type=int, default=100, help="early stop if maximum number of epochs without validation loss decrease reached")
    parser.add_argument("--verbose", type=bool, default=False)
    
    # Evaluation
    parser.add_argument("--recall-at-5", type=bool, default=True, help="whether to evaluate with recall@5")
    parser.add_argument("--mrr", type=bool, default=True, help="whether to evaluate with mean reciprocal rank")
    
    args = parser.parse_args()
    validate_arguments(args)
    return args


def validate_arguments(args):
    assert os.path.exists(args.labels_path)
    if args.src_repr_model in ["BOW", "TFIDF", "content2vec", "content2vec_skill2vec", "TAMF"]:
        assert os.path.exists(args.src_problems_path)
    if args.dst_repr_model in ["BOW", "TFIDF", "content2vec", "content2vec_skill2vec", "TAMF"]:
        assert os.path.exists(args.dst_problems_path)
    if args.src_repr_model in ["skill2vec", "content2vec_skill2vec", "TAMF"]:
        assert os.path.exists(args.src_sequences_path)
        assert os.path.exists(args.src_skill2id_path)
    if args.dst_repr_model in ["skill2vec", "content2vec_skill2vec", "TAMF"]:
        assert os.path.exists(args.dst_sequences_path)
        assert os.path.exists(args.dst_skill2id_path)
    args.device = torch.device(args.device)
    if not args.expr_name:
        args.expr_name = f"{args.src_name}_{args.src_repr_model}_to_{args.dst_name}_{args.dst_repr_model}"
    args.output_path = os.path.join(args.output_root, args.expr_name)
    os.makedirs(args.output_path, exist_ok=True)
