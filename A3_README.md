# CSCI446/946 Assignment 3 - image analytics with pretrained deep features

This folder is a clone of https://github.com/huggingface/pytorch-image-models, as the spec asks.
Our work is `validate.py` (replaced with the released version, then adapted), `code/` and `data/`.

## Setup

A conda env on Python 3.11, CPU only:

    conda create -n csci946-a3 -c conda-forge --override-channels python=3.11
    conda activate csci946-a3
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r A3_requirements.txt

`data/raw/` and `data/processed/` are not in git (they are shared on OneDrive). `data/raw/` holds:

    data/raw/train_efficientformerv2_s0.snap_dist_in1k.csv     from the "fixed" release (the SharePoint copy is damaged)
    data/raw/val_efficientformerv2_s0.snap_dist_in1k.csv
    data/raw/v2_efficientformerv2_s0.snap_dist_in1k.csv        written by validate.py, see below
    data/raw/{train,val,v2}_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv.zip   main task, as released

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

## Main task: prepared data

`code/02_prepare_data.py` (run once, about 25 minutes) reads the three EVA-02 CSVs in 20,000-row chunks,
stores them as float32 `.npy` arrays, and carves our own validation set out of the training data.
Everything after it loads the arrays, never the 14 GB CSV. Checks are in `data/output/prepare_summary.csv`.

| files in `data/processed/` | rows | use |
|---|---|---|
| `train_small_X/y.npy` | 100,000 (100 per class) | trying models quickly |
| `train_full_X/y.npy` + `split_train_idx.npy` | 1,231,167 | final training |
| `val_X/y.npy` | 50,000 (50 per class, from train) | tuning and model choice |
| `test1_X/y.npy` | 50,000 (ImageNet val) | final evaluation only |
| `test2_X/y.npy` | 10,000 (ImageNetV2) | final evaluation only |

Each set also has a `_meta.csv` with the image path and label per row. Features are 1,024 float32
values per image; labels are 0 to 999. Loading:

    import numpy as np
    X, y = np.load("data/processed/train_small_X.npy"), np.load("data/processed/train_small_y.npy")
    X_val, y_val = np.load("data/processed/val_X.npy"), np.load("data/processed/val_y.npy")

    # the full training set is 5 GB: memory-map it and train in mini-batches (SGD / Adam, Labs 4 and 8)
    X_full = np.load("data/processed/train_full_X.npy", mmap_mode="r")
    y_full = np.load("data/processed/train_full_y.npy")
    idx = np.load("data/processed/split_train_idx.npy")    # excludes our validation rows

Never fit, tune or select on test1 / test2: the spec says that voids the results.
