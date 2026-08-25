# RSNA Knee Abnormality Detection

Multi-label classification of 12 knee MRI findings (ACL, MCL, Medial/Lateral Meniscus,
Medial/Lateral/PF OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture) from
knee MRI studies, for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition.

Current public leaderboard score: **0.841**

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
| `train.py` | training loop, per-fold + `--aggregate-oof` cross-validated scoring |
| `infer.py` | inference, with multi-checkpoint fold-ensembling support |

The four `.py` files are the actual, current source - they're written out and executed
as cells inside `final-submission-rsna.ipynb` on Kaggle, and kept here as standalone
files too so they're easy to read/review/diff without opening the notebook. If you edit
one, keep the notebook's `%%writefile` cell in sync (or vice versa).

## Pipeline

1. **`rsna-preprocessing.ipynb`** - converts raw DICOM studies into a fixed-shape
   cache: 6 anatomical "slots" per study x 3 depth-slices per slot (stacked as
   fake-RGB to match pretrained image backbones), 130mm physical crop, resized to
   336px, with laterality derived geometrically from image center rather than DICOM
   corner metadata.

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
   for fold ensembling.

7. **`final-submission-rsna.ipynb`** - the notebook actually submitted to the
   competition. Writes out `dataset.py`/`model.py`/`train.py`/`infer.py`, trains all 5
   folds (each committed separately, then mounted back in as a Kaggle input to avoid
   retraining), runs `--aggregate-oof` as a sanity check, and produces the final
   ensembled `submission.csv`.

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
- Weakest labels by cross-validated AUC: Medial Meniscus (0.68), MCL (0.72), Lateral
  Meniscus (0.72) - likely a genuine visual-difficulty limitation of the current
  preprocessing (only 3 depth-slices per slot) rather than a label-quality issue, since
  MCL has one of the highest silver-label accuracies (96.6%) yet one of the lowest
  model AUCs.
- Kaggle's `/kaggle/working` is ephemeral per interactive session - trained checkpoints
  only survive if you commit ("Save Version -> Save & Run All") and then mount that
  committed version as an input to later sessions.

## Future plans

- **Swap DINOv2 for a RadImageNet-pretrained backbone.** DINOv2 was pretrained on
  natural photos, not medical imaging - RadImageNet (1.35M annotated CT/MRI/ultrasound
  images) has published results beating ImageNet-style pretraining on radiology
  transfer tasks, and should be a closer starting point for knee MRI specifically.
- **Grouped (not random) k-fold, grouped by scanner/site if identifiable.** Competition
  discussion has documented that random k-fold on this dataset inflates AUC by ~0.05
  through scanner memorization - the model learning to recognize which machine/site
  produced a study rather than the actual finding. Worth checking whether `build_splits`
  needs to group by something more than shared report text to get an honest CV number.
- **Cross-architecture ensembling**, not just cross-fold. Right now all 5 folds share
  the same DINOv2 architecture; ensembling predictions from a genuinely different
  backbone (e.g. RadImageNet-ResNet50 alongside DINOv2) tends to reduce correlated
  errors more than ensembling 5 copies of the same architecture.
- **Prompt caching in `llm-api-labeler.ipynb`.** The system prompt is repeated on every
  API call across ~4,349 reports - Anthropic's prompt caching would cut labeling cost
  without changing any output, worth doing if the labeler gets rerun.
