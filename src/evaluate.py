import os
import numpy as np


def evaluate(args, clf, X, y):
    # Get predictions
    preds = clf.predict(X)

    preds, labels = np.array(preds), np.array(y)
    n_correct = (preds == labels).astype(int).sum()

    # Precision, Recall
    eq = preds == labels
    neq = preds != labels

    pos_preds = preds == 1
    neg_preds = preds == 0

    tp = np.where(pos_preds, eq, 0).astype(int).sum()
    fp = np.where(pos_preds, neq, 0).astype(int).sum()
    fn = np.where(neg_preds, neq, 0).astype(int).sum()
    
    results = {
            'tp': tp,
            'fp': fp,
            'fn': fn
            }

    F, precision, recall = F_precision_recall(tp, fp, fn)
    metrics = {'F': F, 'precision': precision, 'recall': recall}

    return results, metrics


def F_precision_recall(tp, fp, fn):
    if tp + fp > 0.:
        precision = tp / (tp + fp)
    else:
        precision = 0.

    if tp + fn > 0.:
        recall = tp / (tp + fn)
    else:
        recall = 0.

    if precision + recall > 0.:
        F = (2 * precision * recall) / (precision + recall)
    else:
        F = 0.

    return F, precision, recall


def micro_average(results):
    tp, fp, fn = 0, 0, 0
    for v in results.values():
        tp += v['tp']
        fp += v['fp']
        fn += v['fn']
    
    return F_precision_recall(tp, fp, fn)

