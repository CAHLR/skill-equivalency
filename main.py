import os
import json
from arguments import parser
import translate
import evaluate
from skill_representations import BOW, TFIDF, content2vec, skill2vec, content2vec_skill2vec, TAMF


def main():
    args = parser()
    print("=" * 20)
    print("Experiment parameters:")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("=" * 20)
    src_repr_path = os.path.join(args.output_path, f"{args.src_repr_model}_{args.src_name}.csv")
    dst_repr_path = os.path.join(args.output_path, f"{args.dst_repr_model}_{args.dst_name}.csv")
    if args.src_repr_model == args.dst_repr_model:
        if args.src_repr_model == "BOW":
            print("Generating BOW representations for source and destination")
            BOW.generate_BOW_vectors_two_taxonomies(
                args.src_problems_path, args.dst_problems_path,
                src_repr_path, dst_repr_path)
            results = evaluate.evaluate(src_repr_path, dst_repr_path, args.labels_path, args.recall_at_5, args.mrr)
        elif args.src_repr_model == "TFIDF":
            print("Generating TFIDF representations for source and destination")
            TFIDF.generate_TFIDF_vectors_two_taxonomies(
                args.src_problems_path, args.dst_problems_path,
                src_repr_path, dst_repr_path)
            results = evaluate.evaluate(src_repr_path, dst_repr_path, args.labels_path, args.recall_at_5, args.mrr)
        elif args.src_repr_model == "content2vec":
            print("Generating content2vec representations for source and destination")
            content2vec.generate_content2vec_vectors_two_taxonomies(
                args.src_problems_path, args.dst_problems_path,
                src_repr_path, dst_repr_path,
                args.content2vec_size, args.content2vec_window, args.content2vec_min_count)
            results = evaluate.evaluate(src_repr_path, dst_repr_path, args.labels_path, args.recall_at_5, args.mrr)
        elif args.src_repr_model == "skill2vec":
            print("Generating skill2vec representations for source and destination")
            skill2vec.generate_skill2vec_vectors_two_taxonomies(
                args.src_sequences_path, args.src_skill2id_path, src_repr_path,
                args.src_skill2vec_size, args.src_skill2vec_window, 
                args.dst_sequences_path, args.dst_skill2id_path, dst_repr_path,
                args.dst_skill2vec_size, args.dst_skill2vec_window)
            print("Learning skill translation")
            training_labels_path, test_labels_path = translate.split_labels(args.labels_path)
            src_translated_path = translate.generate_translated_source_vectors(
                src_repr_path, dst_repr_path, training_labels_path, args.device,
                args.num_epochs, args.val_ratio, args.max_val_non_decreasing_epochs, args.verbose)
            results = evaluate.evaluate(src_translated_path, dst_repr_path, test_labels_path, args.recall_at_5, args.mrr)
        elif args.src_repr_model == "content2vec_skill2vec":
            src_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.src_name}.csv")
            src_skill2vec_path = os.path.join(args.output_path, f"skill2vec_{args.src_name}.csv")
            dst_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.dst_name}.csv")
            dst_skill2vec_path = os.path.join(args.output_path, f"skill2vec_{args.dst_name}.csv")
            print("Generating content2vec representations for source and destination")
            content2vec.generate_content2vec_vectors_two_taxonomies(
                args.src_problems_path, args.dst_problems_path,
                src_content2vec_path, dst_content2vec_path,
                args.content2vec_size, args.content2vec_window, args.content2vec_min_count)
            print("Generating skill2vec representations for source and destination")
            skill2vec.generate_skill2vec_vectors_two_taxonomies(
                args.src_sequences_path, args.src_skill2id_path, src_skill2vec_path,
                args.src_skill2vec_size, args.src_skill2vec_window, 
                args.dst_sequences_path, args.dst_skill2id_path, dst_skill2vec_path,
                args.dst_skill2vec_size, args.dst_skill2vec_window)
            print("Learning skill translation")
            training_labels_path, test_labels_path = translate.split_labels(args.labels_path)
            src_translated_path = translate.generate_translated_source_vectors(
                src_skill2vec_path, dst_skill2vec_path, training_labels_path, args.device,
                args.num_epochs, args.val_ratio, args.max_val_non_decreasing_epochs, args.verbose)
            print("Combining content2vec and skill2vec representations for source and destination")
            content2vec_skill2vec.combine_content2vec_and_skill2vec_two_taxonomies(
                src_content2vec_path, src_translated_path, src_repr_path,
                dst_content2vec_path, dst_skill2vec_path, dst_repr_path)
            results = evaluate.evaluate(src_repr_path, dst_repr_path, test_labels_path, args.recall_at_5, args.mrr)
        elif args.src_repr_model == "TAMF":
            src_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.src_name}.csv")
            dst_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.dst_name}.csv")
            print("Generating content2vec representations for source and destination")
            content2vec.generate_content2vec_vectors_two_taxonomies(
                args.src_problems_path, args.dst_problems_path,
                src_content2vec_path, dst_content2vec_path,
                args.content2vec_size, args.content2vec_window, args.content2vec_min_count)
            print("Generating TAMF representations for source and destination")
            TAMF.generate_TAMF_vectors_two_taxonomies(
                args.src_sequences_path, args.src_skill2id_path, src_content2vec_path, src_repr_path,
                args.src_TAMF_k, args.src_TAMF_lambda, 
                args.dst_sequences_path, args.dst_skill2id_path, dst_content2vec_path, dst_repr_path,
                args.dst_TAMF_k, args.dst_TAMF_lambda)
            print("Learning skill translation")
            training_labels_path, test_labels_path = translate.split_labels(args.labels_path)
            src_translated_path = translate.generate_translated_source_vectors(
                src_repr_path, dst_repr_path, training_labels_path, args.device,
                args.num_epochs, args.val_ratio, args.max_val_non_decreasing_epochs, args.verbose)
            results = evaluate.evaluate(src_translated_path, dst_repr_path, test_labels_path, args.recall_at_5, args.mrr)
    elif args.src_repr_model != args.dst_repr_model:
        ## Source
        if args.src_repr_model == "BOW":
            print("Generating BOW representations for source")
            BOW.generate_BOW_vectors_one_taxonomy(args.src_problems_path, src_repr_path)
        elif args.src_repr_model == "TFIDF":
            print("Generating TFIDF representations for source")
            TFIDF.generate_TFIDF_vectors_one_taxonomy(args.src_problems_path, src_repr_path)
        elif args.src_repr_model == "content2vec":
            print("Generating content2vec representations for source")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.src_problems_path, src_repr_path,
                args.src_content2vec_size, args.src_content2vec_window, args.src_content2vec_min_count)
        elif args.src_repr_model == "skill2vec":
            print("Generating skill2vec representations for source")
            skill2vec.generate_skill2vec_vectors_one_taxonomy(
                args.src_sequences_path, args.src_skill2id_path, src_repr_path,
                args.src_skill2vec_size, args.src_skill2vec_window)
        elif args.src_repr_model == "content2vec_skill2vec":
            src_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.src_name}.csv")
            src_skill2vec_path = os.path.join(args.output_path, f"skill2vec_{args.src_name}.csv")
            print("Generating content2vec representations for source")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.src_problems_path, src_content2vec_path,
                args.src_content2vec_size, args.src_content2vec_window, args.src_content2vec_min_count)
            print("Generating skill2vec representations for source")
            skill2vec.generate_skill2vec_vectors_one_taxonomy(
                args.src_sequences_path, args.src_skill2id_path, src_skill2vec_path,
                args.src_skill2vec_size, args.src_skill2vec_window)
            print("Combining content2vec and skill2vec representations for source")
            content2vec_skill2vec.combine_content2vec_and_skill2vec_one_taxonomy(
                src_content2vec_path, src_skill2vec_path, src_repr_path)
        elif args.src_repr_model == "TAMF":
            src_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.src_name}.csv")
            print("Generating content2vec representations for source")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.src_problems_path, src_content2vec_path,
                args.src_content2vec_size, args.src_content2vec_window, args.src_content2vec_min_count)
            print("Generating TAMF representations for source")
            TAMF.generate_TAMF_vectors_one_taxonomy(
                args.src_sequences_path, args.src_skill2id_path, src_content2vec_path,
                src_repr_path, args.src_TAMF_k, args.src_TAMF_lambda)
        ## Destination
        if args.dst_repr_model == "BOW":
            print("Generating BOW representations for destination")
            BOW.generate_BOW_vectors_one_taxonomy(args.dst_problems_path, dst_repr_path)
        elif args.dst_repr_model == "TFIDF":
            print("Generating TFIDF representations for destination")
            TFIDF.generate_TFIDF_vectors_one_taxonomy(args.dst_problems_path, dst_repr_path)
        elif args.dst_repr_model == "content2vec":
            print("Generating content2vec representations for destination")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.dst_problems_path, dst_repr_path,
                args.dst_content2vec_size, args.dst_content2vec_window, args.dst_content2vec_min_count)
        elif args.dst_repr_model == "skill2vec":
            print("Generating skill2vec representations for destination")
            skill2vec.generate_skill2vec_vectors_one_taxonomy(
                args.dst_sequences_path, args.dst_skill2id_path, dst_repr_path,
                args.dst_skill2vec_size, args.dst_skill2vec_window)
        elif args.dst_repr_model == "content2vec_skill2vec":
            dst_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.dst_name}.csv")
            dst_skill2vec_path = os.path.join(args.output_path, f"skill2vec_{args.dst_name}.csv")
            print("Generating content2vec representations for destination")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.dst_problems_path, dst_content2vec_path,
                args.dst_content2vec_size, args.dst_content2vec_window, args.dst_content2vec_min_count)
            print("Generating skill2vec representations for destination")
            skill2vec.generate_skill2vec_vectors_one_taxonomy(
                args.dst_sequences_path, args.dst_skill2id_path, dst_skill2vec_path,
                args.dst_skill2vec_size, args.dst_skill2vec_window)
            print("Combining content2vec and skill2vec representations for destination")
            content2vec_skill2vec.combine_content2vec_and_skill2vec_one_taxonomy(
                dst_content2vec_path, dst_skill2vec_path, dst_repr_path)
        elif args.dst_repr_model == "TAMF":
            dst_content2vec_path = os.path.join(args.output_path, f"content2vec_{args.dst_name}.csv")
            print("Generating content2vec representations for destination")
            content2vec.generate_content2vec_vectors_one_taxonomy(
                args.dst_problems_path, dst_content2vec_path,
                args.dst_content2vec_size, args.dst_content2vec_window, args.dst_content2vec_min_count)
            print("Generating TAMF representations for destination")
            TAMF.generate_TAMF_vectors_one_taxonomy(
                args.dst_sequences_path, args.dst_skill2id_path, dst_content2vec_path,
                dst_repr_path, args.dst_TAMF_k, args.dst_TAMF_lambda)
        print("Learning skill translation")
        training_labels_path, test_labels_path = translate.split_labels(args.labels_path)
        src_translated_path = translate.generate_translated_source_vectors(
            src_repr_path, dst_repr_path, training_labels_path, args.device,
            args.num_epochs, args.val_ratio, args.max_val_non_decreasing_epochs, args.verbose)
        results = evaluate.evaluate(src_translated_path, dst_repr_path, test_labels_path, args.recall_at_5, args.mrr)
    print("=" * 20)
    print(f"results {results}")
    json.dump(results, open(os.path.join(args.output_path, "results.json"), "w"))


if __name__ == "__main__":
    main()
