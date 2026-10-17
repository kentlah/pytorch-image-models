from pathlib import Path
import numpy as np
import pandas as pd

root = Path(".")
df = pd.read_csv(root / "data/output/classifier_a_test1_overlap.csv")
matched = df.loc[df["first_matching_train_row"] >= 0].copy()

y_train = np.load(root / "data/processed/train_full_y.npy")
indices = matched["first_matching_train_row"].to_numpy(dtype=np.int64)

matched["matching_train_label"] = y_train[indices]
matched["labels_agree"] = (
    matched["true_label"] == matched["matching_train_label"]
)
matched["prediction_matches_train_label"] = (
    matched["predicted_label"] == matched["matching_train_label"]
)

matched.to_csv(
    root / "data/output/classifier_a_overlap_label_check.csv",
    index=False,
)

print("Matching test examples:", len(matched))
print("Same label:", int(matched["labels_agree"].sum()))
print("Different label:", int((~matched["labels_agree"]).sum()))
print(
    "Prediction matches first matching training label:",
    int(matched["prediction_matches_train_label"].sum()),
)