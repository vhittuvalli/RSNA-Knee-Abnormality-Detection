# RSNA Knee Abnormality Detection

Multi-label classification of 12 knee MRI findings (ACL, MCL, Medial/Lateral Meniscus,
Medial/Lateral/PF OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture) from
knee MRI studies, for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition.

Current public leaderboard score: **0.841** (single fold, DINOv2, 9 depth-slices/slot +
meniscus-specific crop + TTA - see [Key results](#key-results--lessons)). This score
predates the 5-fold ensemble described below; `final-submission-rsna.ipynb` now
combines both improvements (5-fold ensembling + the current preprocessing/TTA), which
should push past 0.841 but hasn't been submitted yet.

## Repo layout

Three Kaggle notebooks (exported as `.ipynb`) plus standalone `.py` mirrors of the
training/inference code for readability:

| File | What it is |
|---|---|
| `rsna-preprocessing.ipynb` | DICOM -> image cache preprocessing |
| `llm-api-labeler.ipynb` | LLM-based soft labeling of unlabeled reports |
| `final-submission-rsna.ipynb` | training + inference + submission - writes `dataset.py`, `model.py`, `train.py`, `infer.py` via `%%writefile` cells, then runs them |
| `dataset.py` | PyTorch `Dataset` / cross-fit split builder |
| `model.py` | `KneeMRIModel` architecture |
| `train.py` | training loop, per-fold + `--aggregate-oof` cross-validated scoring; supports `resnet50`, `radimagenet_resnet50`, and `dinov2` backbones |
| `infer.py` | inference, with multi-checkpoint fold-ensembling support and `--tta` (rotation test-time augmentation) |

The four `.py` files are the actual, current source - they're written out and executed
as cells inside `final-submission-rsna.ipynb` on Kaggle, and kept here as standalone
files too so they're easy to read/review/diff without opening the notebook. If you edit
one, keep the notebook's `%%writefile` cell in sync (or vice versa).

## Pipeline

1. **`rsna-preprocessing.ipynb`** - converts raw DICOM studies into a fixed-shape
   cache: 6 anatomical "slots" per study x 9 depth-slices per slot (split into 3
   non-overlapping windows of 3 slices each at train/infer time, stacked as fake-RGB
   to match pretrained image backbones - see `dataset.py`'s `window_slots`), 130mm
   physical crop (tightened to 90mm for the meniscus-specific `SAG_FLUID_FS` slot,
   via `CROP_MM_BY_SLOT`), resized to 224px, with laterality derived geometrically
   from image center rather than DICOM corner metadata.
   **NOTE:** this notebook must be re-run and its output re-attached before
   retraining - its committed output still reflects the older 3-slice/336px
   settings until that happens (see `final-submission-rsna.ipynb`'s path-config
   cell).

2. **`llm-api-labeler.ipynb`** - most of the training set (4,349 of 4,407 studies) has
   a free-text radiology report but no structured labels. This uses an LLM to extract
   a 4-way status (affirmed / hedged / negated / absent) per finding per report,
   handling multilingual reports and negation direction, then maps status to a soft
   numeric target (e.g. affirmed=1.0, negated=0.1, absent=0.28, hedged=per-label
   calibrated value) rather than collapsing to hard 0/1. Validated against the 58 gold
   radiologist-labeled studies (86.5% agreement).

3. **`dataset.py`** - loads the preprocessed image cache and merged gold+silver labels,
   builds a 5-fold cross-fit split of the gold-labeled studies for validation, and
   weights each training example's loss contribution by that label's silver-label
   confidence (gold examples get full weight).

4. **`model.py`** - `KneeMRIModel`: a DINOv2 (or ResNet50) backbone shared across all 6
   slots, mean+max pooled across slots, followed by an MLP classification head over the
   12 labels.

5. **`train.py`** - trains one fold at a time (`--gold-fold 0..4`), tracking gold-only
   and silver-only validation AUC per epoch, early stopping, and writing out-of-fold
   gold predictions for later cross-validated scoring via `--aggregate-oof`.

6. **`infer.py`** - runs inference over the test set, optionally averaging predictions
   across multiple fold checkpoints (`--checkpoint fold0/best.pt fold1/best.pt ...`)
   for fold ensembling, and optionally averaging over 3 small-angle rotations per
   study via `--tta` (matching `train.py`'s own augmentation range) - both stack on
   top of each other with no retraining required for the TTA half.

7. **`final-submission-rsna.ipynb`** - the notebook actually submitted to the
   competition. Writes out `dataset.py`/`model.py`/`train.py`/`infer.py`, trains all 5
   folds (each committed separately, then mounted back in as a Kaggle input to avoid
   retraining), runs `--aggregate-oof` as a sanity check, rebuilds the test cache live
   with `--cache-slices 9`, and produces the final `submission.csv` ensembled across
   all 5 fold checkpoints with `--tta` enabled.

## Key results / lessons

- Soft/graded labels instead of hard 0/1 for LLM-derived silver labels improved
  validation accuracy from 80.7% (regex baseline) to 86.5%.
- `roc_auc_score` requires binary ground truth - AUC evaluation against soft targets
  must binarize at a 0.5 threshold, or validation silently produces `NaN` and the run
  never saves a checkpoint.
- A learned attention-pooling layer was tried as a third pooling signal alongside
  mean/max, but caused a real regression (0.8182 -> 0.7684 AUC) because its parameters
  were never added to the optimizer's parameter groups, so it trained on frozen random
  weights. Reverted to mean+max pooling only.
- 5-fold ensembling (average `sigmoid` predictions across fold checkpoints) gave a
  measurable public leaderboard improvement (0.791 -> 0.821) with no architecture
  changes.
- Weakest labels by cross-validated AUC (against the old 3-slice/336px preprocessing):
  Medial Meniscus (0.68), MCL (0.72), Lateral Meniscus (0.72) - likely a genuine
  visual-difficulty limitation of only having 3 depth-slices per slot rather than a
  label-quality issue, since MCL has one of the highest silver-label accuracies (96.6%)
  yet one of the lowest model AUCs.
- Raising depth-slices per slot from 3 to 9 (windowed into 3 fake-RGB triplets per
  slot at train/infer time), adding a meniscus-specific 90mm crop on the
  `SAG_FLUID_FS` slot, and adding 3-angle rotation TTA at inference together lifted a
  *single, un-ensembled* fold to 0.841 on the public leaderboard - already above the
  old 5-fold-ensembled score (0.821) on the previous preprocessing, without needing
  fold ensembling at all. This is consistent with the "more depth-slices per slot"
  hypothesis above: the weakest labels were plausibly slice-starved, not
  label-starved.
- Kaggle's `/kaggle/working` is ephemeral per interactive session - trained checkpoints
  only survive if you commit ("Save Version -> Save & Run All") and then mount that
  committed version as an input to later sessions.

## Future plans

- **Actually train and ensemble the RadImageNet backbone.** `model.py`/`train.py`/
  `infer.py` already support `--backbone radimagenet_resnet50` (a Keras -> ONNX ->
  PyTorch conversion of a ResNet50 pretrained on 1.35M CT/MRI/ultrasound images), but
  no submission has actually used it yet - the 0.841 score is DINOv2-only. Training a
  RadImageNet fold set and averaging its predictions with the DINOv2 folds
  (cross-architecture ensembling, not just cross-fold) tends to reduce correlated
  errors more than ensembling 5 copies of the same architecture.
- **Grouped (not random) k-fold, grouped by scanner/site if identifiable.** Competition
  discussion has documented that random k-fold on this dataset inflates AUC by ~0.05
  through scanner memorization - the model learning to recognize which machine/site
  produced a study rather than the actual finding. Worth checking whether `build_splits`
  needs to group by something more than shared report text to get an honest CV number.
- **Confirm the 5-fold + new-preprocessing combination actually stacks.** The 0.841
  score and the 0.791 -> 0.821 ensembling gain were measured independently (different
  preprocessing, different fold counts) - `final-submission-rsna.ipynb` now runs both
  together, but that combined score hasn't been submitted/confirmed yet.
- **Prompt caching in `llm-api-labeler.ipynb`.** The system prompt is repeated on every
  API call across ~4,349 reports - Anthropic's prompt caching would cut labeling cost
  without changing any output, worth doing if the labeler gets rerun.
