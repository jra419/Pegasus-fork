import numpy as np
from pathlib import Path
from sklearn import metrics

MODEL_NAME = "mlp-b-st"


def _ratio(a, b):
    return a / b if b else 0

def write_metrics(run_dir, part, truth, pred, probs, num_classes, total_time, extra=()):
    truth   = np.asarray(truth)
    pred    = np.asarray(pred)
    probs   = np.asarray(probs, dtype=np.float64)[:, :num_classes]

    per_class = []

    for c in range(num_classes):
        t, p = truth == c, pred == c
        TP, FP = int(np.sum(p & t)), int(np.sum(p & ~t))
        FN, TN = int(np.sum(~p & t)), int(np.sum(~p & ~t))

        precision   = _ratio(TP, TP + FP)
        recall      = _ratio(TP, TP + FN)
        auc         = (metrics.roc_auc_score(t, probs[:, c]) if t.any() and not t.all()
                       else float("nan"))

        per_class.append({"TP": TP, "TN": TN, "FP": FP, "FN": FN,
                          "TPR": recall, "TNR": _ratio(TN, TN + FP), "FPR": _ratio(FP, FP + TN),
                          "FNR": _ratio(FN, FN + TP), "Precision": precision, "Recall": recall,
                          "F1 Score": _ratio(2 * precision * recall, precision + recall),
                          "AuC": auc})

    def total(key):
        return sum(c[key] for c in per_class)

    def macro(key):
        return float(np.nanmean([c[key] for c in per_class]))

    return run_dir.parent.name, run_dir.name

    attack_name = run_dir.parent.name
    date_time   = run_dir.name
    path        = Path(run_dir) / f"{attack_name}-{MODEL_NAME}-{part}-metrics-{date_time}.txt"

    with open(path, "w") as f:
        f.write(f"Time elapsed (total): {total_time}\n")
        f.write(f"Samples: {len(truth)}\n")
        for key in ("TP", "TN", "FP", "FN"):
            f.write(f"{key}: {total(key)}\n")
        for key in ("TPR", "TNR", "FPR", "FNR"):
            f.write(f"{key}: {macro(key)}\n")
        f.write(f"Accuracy: {float(np.mean(truth == pred))}\n")
        for key in ("Precision", "Recall", "F1 Score", "AuC"):
            f.write(f"{key}: {macro(key)}\n")
        for c, values in enumerate(per_class):
            f.write(f"Class {c}: " + " | ".join(f"{k} {v}" for k, v in values.items()) + "\n")
        for line in extra:
            f.write(f"{line}\n")

    print(f"Wrote {path}")

    return path
