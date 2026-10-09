# 01_practice_classifier.py - practice task: a classifier on EfficientFormerV2-S0 features (176-d)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
MODEL = "efficientformerv2_s0.snap_dist_in1k"
PER_CLASS = 200           # training images per class; the full 1.27 M rows do not fit lbfgs in memory


# 1. Load
# train and val are the supplied files; v2 was written by validate.py from the 10,000 ImageNetV2 images
def load(name):
    header = pd.read_csv(RAW / ("%s_%s.csv" % (name, MODEL)), nrows=0).columns
    features = [c for c in header if c.isdigit()]
    df = pd.read_csv(RAW / ("%s_%s.csv" % (name, MODEL)), usecols=["label"] + features,
                     dtype=dict({c: np.float32 for c in features}, label=np.int16))
    return df[features].to_numpy(), df["label"].to_numpy()


X_train_all, y_train_all = load("train")
X_val, y_val = load("val")
X_v2, y_v2 = load("v2")
print("train:", X_train_all.shape, "val (test set 1):", X_val.shape, "v2 (test set 2):", X_v2.shape)
print("images per class in train: %d to %d" % (np.bincount(y_train_all).min(), np.bincount(y_train_all).max()))


# 2. Train
# a class-stratified sample keeps every class equally represented
rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.flatnonzero(y_train_all == c), PER_CLASS, replace=False)
                      for c in range(1000)])
X_train, y_train = X_train_all[idx], y_train_all[idx]
del X_train_all
print("training sample:", X_train.shape)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=SEED))
model.fit(X_train, y_train)


# 3. Evaluate on both test sets
def top_k(X, y, k):
    top = model.classes_[np.argsort(-model.predict_proba(X), axis=1)[:, :k]]
    return (top == y[:, None]).any(axis=1).mean()


results = pd.DataFrame({
    "test_set": ["test set 1 (ImageNet val)", "test set 2 (ImageNetV2)"],
    "rows": [len(y_val), len(y_v2)],
    "top1": [top_k(X_val, y_val, 1), top_k(X_v2, y_v2, 1)],
    "top5": [top_k(X_val, y_val, 5), top_k(X_v2, y_v2, 5)],
})
results.to_csv(OUT / "practice_results.csv", index=False)
print(results.round(4).to_string(index=False))
print("top-1 gap (test set 1 - test set 2): %.1f points" % (100 * (results.top1[0] - results.top1[1])))

fig, ax = plt.subplots(figsize=(6.2, 3))
x = np.arange(2)
ax.bar(x - 0.2, results.top1, 0.4, label="top-1", color="#2a78d6")
ax.bar(x + 0.2, results.top5, 0.4, label="top-5", color="#eb6834")
for i in range(2):
    ax.text(i - 0.2, results.top1[i] + 0.01, "%.1f%%" % (100 * results.top1[i]), ha="center")
    ax.text(i + 0.2, results.top5[i] + 0.01, "%.1f%%" % (100 * results.top5[i]), ha="center")
ax.set_xticks(x)
ax.set_xticklabels(results.test_set)
ax.set_ylim(0, 1.15)
ax.set_ylabel("accuracy")
ax.set_title("Logistic regression on EfficientFormerV2-S0 features")
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUT / "fig_practice_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
