import os
import numpy as np

def evaluate(args, clf, X_test, y_test):
    if args.model_type == 'majority':
        preds = clf.predict(X_test)

    preds, labels = np.array(preds), np.array(y_test)
    n_correct = (preds == labels).astype(int).sum()

    # Precision, Recall
    eq = preds == labels
    neq = preds != labels

    pos_preds = preds == 1
    neg_preds = preds == 0

    tp = np.where(pos_preds, eq, 0).astype(int).sum()
    fp = np.where(pos_preds, neq, 0).astype(int).sum()
    fn = np.where(neg_preds, neq, 0).astype(int).sum()
    
    #tp_i = np.where(neg_preds, eq, 0).astype(int).sum()
    #fp_i = np.where(neg_preds, neq, 0).astype(int).sum()
    #fn_i = np.where(pos_preds, neq, 0).astype(int).sum()

    results = {
            'tp': tp,
            'fp': fp,
            'fn': fn
            }

    return results
