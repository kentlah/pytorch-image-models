# CSCI446/946 Assignment 3 - image analytics with pretrained deep features

This folder is a clone of https://github.com/huggingface/pytorch-image-models, as the spec asks.
Our work is `validate.py` (replaced with the released version, then adapted), `code/` and `data/`.

## Setup

Python 3.8 (the group's `csci946env`), CPU only:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install pyyaml huggingface_hub safetensors pandas scikit-learn matplotlib

`data/raw/` is not in git (2.6 GB). Put the extracted CSVs there:

    data/raw/train_efficientformerv2_s0.snap_dist_in1k.csv     from the "fixed" release (the SharePoint copy is damaged)
    data/raw/val_efficientformerv2_s0.snap_dist_in1k.csv
    data/raw/v2_efficientformerv2_s0.snap_dist_in1k.csv        written by validate.py, see below

## Practice task

1. `validate.py` is the released file with one change: it saves the 176-d features before the head
   (`pre_logits`) instead of the 8,624-d feature map, adds each image's path, and writes
   `data/raw/v2_<model>.csv`. Run from this folder:

       python validate.py <path>/imagenetv2-matched-frequency-format-val --device cpu --model efficientformerv2_s0.snap_dist_in1k

   The model's own head scores 64.27% top-1 / 85.51% top-5 on ImageNetV2.

2. `code/01_practice_classifier.py` trains logistic regression on 200 training images per class
   (SEED = 7) and scores it on the supplied val features (test set 1) and our v2 features (test set 2).
   Writes `data/output/practice_results.csv` and `fig_practice_accuracy.png`.
   To run without a display: `MPLBACKEND=Agg python code/01_practice_classifier.py`.

   Result (`data/output/practice_results.csv`):

   | test set | rows | top-1 | top-5 |
   |---|---|---|---|
   | test set 1 (ImageNet val) | 50,000 | 69.3% | 88.8% |
   | test set 2 (ImageNetV2) | 10,000 | 57.0% | 79.7% |

   A 12.2-point top-1 gap, in line with Recht et al. lbfgs stops at `max_iter=300` before
   converging, so these are slight underestimates. The fit takes about 2.5 hours on a laptop CPU.
