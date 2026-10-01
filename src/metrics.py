import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, confusion_matrix

def compute_detection_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    pred = p >= .5; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    negatives = p[y == 0]; positives = p[y == 1]
    cutoff = np.quantile(negatives, .99) if len(negatives) else 1.0
    return {"auroc": float(roc_auc_score(y,p)) if len(np.unique(y)) > 1 else float("nan"), "auprc": float(average_precision_score(y,p)), "tpr_at_1pct_fpr": float(np.mean(positives >= cutoff)) if len(positives) else float("nan"), "fnr": float(fn/max(fn+tp,1))}

def compute_calibration_metrics(labels, probabilities, bins=10):
    y, p = np.asarray(labels), np.asarray(probabilities); edges = np.linspace(0,1,bins+1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if np.any(mask): ece += np.mean(mask) * abs(np.mean(y[mask])-np.mean(p[mask]))
    return {"ece": float(ece), "brier": float(np.mean((p-y)**2)), "nll": float(log_loss(y,p,labels=[0,1]))}

def compute_false_negative_confidence(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities); vals = p[(y == 1) & (p < .5)]
    return {"count": int(len(vals)), "mean": float(np.mean(vals)) if len(vals) else float("nan"), "max": float(np.max(vals)) if len(vals) else float("nan")}

