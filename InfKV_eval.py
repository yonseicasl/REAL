import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    'comprehension_and_reasoning': qa_f1_score,
    'computation': qa_f1_score,
    'multiple_information_retrieval': qa_f1_score,
    'timeline_reorder': qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_score,
    "trec": qa_f1_score,
    "triviaqa": qa_f1_score,
    "dureader":qa_f1_zh_score,
    "samsum": rouge_score,
    "lsht":classification_score,
    "passage_count":count_score,
    "passage_retrieval_en":retrieval_score,
    "passage_retrieval_zh":retrieval_zh_score,
    "lcc":retrieval_zh_score,
    "repobench-p":code_sim_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='meta-llama-3-8b-instruct')
    parser.add_argument('--capacity', type=int, default=128)
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    args.results_dir = f"{args.results_dir}/{args.model}_{args.capacity}"
    dataset_list = [
    "narrativeqa"
    # "qasper",
    # "multifieldqa_en",
    # "multifieldqa_zh",
    # "hotpotqa",
    # "2wikimqa",
    # "musique",
    # "dureader",
    # "gov_report",
    # "qmsum",
    # "multi_news",
    # "vcsum",
    # "trec",
    # "triviaqa",
    # "samsum",
    # "lsht",
    # "passage_count",
    # "passage_retrieval_en",
    # "passage_retrieval_zh",
    # "lcc",
    # "repobench-p",
    # "comprehension_and_reasoning",
    # "multiple_information_retrieval",
    # "timeline_reorder",
    # "computation"
        ]
    
    results_list = [
        ["dataset"],
        ["InfKV"],
    ]
    total_scores = []

    for dataset in dataset_list:
        
        results_list[0].append(dataset)
        
        for idx, method in enumerate(["InfKV"]):
            try:
                dataset_results_dir = os.path.join(args.results_dir, dataset)

                scores = dict()
                predictions, answers, lengths = [], [], []
                all_classes = None
                
                if not os.path.isdir(dataset_results_dir):
                    print(f"Directory not found: {dataset_results_dir}")
                    raise FileNotFoundError

                found_files = False
                for filename in os.listdir(dataset_results_dir):
                    if filename.endswith(".json") and filename != "metrics.json":
                        filepath = os.path.join(dataset_results_dir, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    predictions.append(data["pred"])
                                    answers.append(data["answers"])
                                    all_classes = data.get("all_classes")
                                    if "length" in data:
                                        lengths.append(data["length"])
                                    found_files = True
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in {filepath}: {e} on line: {line.strip()}")
                                except KeyError as e:
                                    print(f"Missing key {e} in {filepath} for line: {line.strip()}")
                        
                if not found_files:
                     print(f"No result .json files found in {dataset_results_dir}")
                     raise FileNotFoundError

                if args.longbench_e:
                    score = scorer_e(dataset, predictions, answers, lengths, all_classes)
                else:
                    score = scorer(dataset, predictions, answers, all_classes)
                scores[dataset] = score
                    
                output_dir = dataset_results_dir
                
                results_list[idx+1].append(score)
                
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
                total_scores.append(score)
                print(f"dataset {dataset} method {method} scores {scores}")
            except FileNotFoundError:
                results_list[idx+1].append(-1)
                print(f"dataset {dataset} method {method} scores None (Files not found or directory missing)")
                
    import csv
    with open(os.path.join(args.results_dir,f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)
    print(f"Evaluation results saved to: {os.path.join(args.results_dir, 'results.csv')}")


