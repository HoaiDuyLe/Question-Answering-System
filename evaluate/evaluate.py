from __future__ import print_function
from collections import Counter
from pprint import pprint
import string
import re
import os
import argparse
import json
import sys
import inference

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth.replace('\n', ' '))
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset):
    f1 = exact_match = total = 0
    for paragraph in dataset:
        corpus = paragraph['context'].replace('\n', ' ')
        print(corpus)
        for qa in paragraph['qas']:
            total += 1
            question = qa['question']
            answer = inference.infer(corpus,question)
            f1 += metric_max_over_ground_truths(f1_score, answer, qa['answer'])
            exact_match += metric_max_over_ground_truths(exact_match_score, answer, qa['answer'])
            if metric_max_over_ground_truths(f1_score, answer, qa['answer']) != 1.0:
                print("\nQUESTION: {} \nCORRECT_answer: {} \nPREDICTED_answer: {} \nF1 = {}".format \
                (qa['question'], qa['answer'], answer, metric_max_over_ground_truths(f1_score, answer, qa['answer'])))
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Evaluation for ICE')
    parser.add_argument('dataset_file', help='Dataset file')
    args = parser.parse_args()
    with open(args.dataset_file, "r") as dataset_file:
        dataset = json.load(dataset_file, strict = False)
    temp = evaluate(dataset)
    print("\n",temp)