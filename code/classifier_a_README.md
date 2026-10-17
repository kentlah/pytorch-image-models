# Classifier A: regularised linear classification

## Method

Frozen EVA-02 features contain 1,024 values per image across 1,000 classes.
The classifier uses SGDClassifier with logistic loss, an L2 penalty, and
one-versus-rest multiclass classification.

StandardScaler is fitted using training examples only. The shared internal
validation set is used for configuration selection and early stopping.
Neither test set is used for fitting or tuning.

All experiments use seed 7.

## Configuration selection

Three alpha values were compared on the shared 100,000-example training
subset: 1e-05, 0.0001, and 0.001, for five epochs each.

The default learning-rate schedule also depends on alpha, so this comparison
evaluates training configurations rather than isolating regularisation alone.

The leading configuration, alpha=1e-05, was examined over ten subset epochs.

## Full training

The full feature array is memory-mapped. Only rows selected by
split_train_idx.npy are used for fitting, excluding internal validation rows.

Training uses batches of 1,024, a maximum of ten epochs, and early stopping
after three epochs without improved validation top-1 accuracy.

The selected checkpoint was epoch 2:
- Internal validation top-1: 95.228%
- Internal validation top-5: 99.656%
- Training examples: 1,231,167
- Total run time, including scaling and validation: approximately 36.8 minutes

The run emitted scikit-learn parallel configuration warnings but completed.
Software versions are recorded in data/output/classifier_a_environment.txt.

## Final test results

| Dataset | Top-1 | Top-5 |
|---|---:|---:|
| ImageNet | 89.856% | 98.988% |
| ImageNetV2 | 82.280% | 96.390% |

The top-1 gap is 7.576 percentage points.

Exported decision scores and margins are not calibrated probabilities.

## Reproduction

Run from the repository root, with the prepared files in data/processed/.

    python code/classifier_a_linear.py --full --epochs 10 --patience 3 --alpha 0.00001
    python code/classifier_a_evaluate.py
    python code/classifier_a_analyse.py
    python code/classifier_a_check_overlap_labels.py

The analysis script also requires the saved five-epoch tuning histories.
The training script refuses to overwrite an existing run's history or model.
Move existing run outputs to a backup location before repeating training.

Datasets and trained model files are excluded from Git.


## Exact-feature overlap sensitivity analysis

Of the 50,000 test1 examples, 773 had feature vectors exactly matching
at least one example in the classifier's training-only split.

Top-1 accuracy was 35.317% for matching examples and 90.712% for
non-matching examples, compared with 89.856% on the complete test1 set.

Excluding matching examples increased the test1–test2 gap from 7.576
to approximately 8.432 percentage points. Thus, these matches do not
explain an inflated test1 score for this classifier.

These are exact feature matches, not verified duplicate image files.
A follow-up check found that all 773 matching test1 examples had a
different label from their first exact-matching training row. The
classifier predicted that training row's label for 439 examples (56.8%).

This identifies conflicting labels attached to identical feature vectors
across the supplied datasets. It does not establish which labels are
correct or whether the original image files are identical. Where multiple
training matches exist, only the first matching row was checked.

The original test scores are retained. The non-matching subset result is
reported separately as a sensitivity analysis. No test-informed relabelling
or training-data removal was performed.


