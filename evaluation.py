
import numpy as np


def relative_scores(scores: dict) -> dict:
    """
    Given a mapping form_name -> log-score, shift so max is 0 (for interpretability).
    """
    max_score = max(scores.values())
    return {k: v - max_score for k, v in scores.items()}


def select_best_form(scores: dict) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def softmax_probabilities(scores: dict) -> dict:
    """
    Softmax over log P(S,F|D) to get relative probabilities.
    """
    log_vals = np.array(list(scores.values()))
    max_log = np.max(log_vals)
    exp_shifted = np.exp(log_vals - max_log)
    probs = exp_shifted / np.sum(exp_shifted)
    return {k: probs[i] for i, k in enumerate(scores.keys())}
