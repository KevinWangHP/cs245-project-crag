import re
import string


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


def calculate_short_answer_EM(generated_answer, gold_answers, language="en"):
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    if language == "en":
        generated_answer = normalize_answer(generated_answer)
    else:
        pass
    match = 0
    for gold_answer in gold_answers:
        gold_answer = normalize_answer(gold_answer)
        if gold_answer in generated_answer:
            match += 1
    return {
        "exact_match": match / len(gold_answers),
        "hit1": match > 0
    }

import evaluate
rouge=evaluate.load("../evaluate_utils/rouge/")
bleu=evaluate.load("../evaluate_utils/bleu/")

def select_candidate(generated_answer, gold_answers):
    max_rouge_score = 0
    candidate_idx = -1
    if isinstance(gold_answers, str):
        return gold_answers
    if len(gold_answers) == 1:
        return gold_answers[0]
    for idx, gold_answer in enumerate(gold_answers):
        rouge_score = rouge.compute(predictions=[generated_answer], references=[gold_answer])
        if rouge_score["rougeL"] > max_rouge_score:
            max_rouge_score = rouge_score["rougeL"]
            candidate_idx = idx
    return gold_answers[candidate_idx]

def mean_rouge(rouge_results):
    rouge_results = {k: sum([rouge_result for rouge_result in rouge_results[k]]) / len(rouge_results["rouge1"]) for k in rouge_results.keys()}
    return {k: round(v * 100, 2) for k, v in rouge_results.items()}
