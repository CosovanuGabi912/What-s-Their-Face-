# Breed classifier: how to reproduce training (issue #6)

ConvNeXt-Tiny (ImageNet weights from torchvision) fine-tuned on Stanford Dogs, 120 breeds.

## Results

- Test split (8,580 images): top-1 accuracy 90.6%, top-3 accuracy 98.5%. Target for Week 7 was 70% and 85%.
- Validation split (2,400 images): top-1 90.4%, top-3 98.1%.
- Best epoch: 13 of 20. Accuracy was already 75% after the first epoch and stopped improving after about epoch 12.
- Training time: 16 minutes on an NVIDIA RTX 5050 Laptop GPU (8 GB), PyTorch 2.11 with CUDA 12.8.

## Calibration (issue #10)

- `scripts/calibrate_classifier.py` fits the temperature on the validation split: T = 0.79. Calibration error drops from 7.7% to 2.5%; accuracy does not change.
- Two thresholds on the calibrated top-1 probability, chosen by trial and error on the validation split with `--sweep` (no F1, never on golden Part A), saved in `threshold.json` and read by `config.py`:
  - top-1 >= 0.82: "confident". 88% of answers; 95% of them are correct.
  - 0.40 <= top-1 < 0.82: "top2", show the two most likely breeds. The true breed is one of the two 92% of the time.
  - top-1 < 0.40: "retry", ask for a clearer photo. Only 1.2% of answers. A higher value mostly sends photos to retry whose breed was already in the top 2 (74% of them at 0.50).
- The sweep also compared the rule "top-1 + top-2 >= threshold" for "top2": at the same share of retries the results were within 1 point, so we keep the simpler top-1 rule used by `shared.schemas.breed_decision`.
- Check on the full test split (8,580 images): 87% of answers are confident and 95.7% of those are correct. 61% of the wrong answers are flagged "not sure"; 92% of the right answers are confident.
- Typical mistake: Siberian Husky vs Eskimo Dog, two very similar breeds. These mistakes usually fall under the threshold.

- Golden Part A (100 photos from the test split, chosen by Part D, #8), with the thresholds above: top-1 88%, top-3 99% (target 70% / 85%). 87% of answers are confident and 94% of those are correct; 58% of the wrong answers are flagged "not sure". Saved in `eval/results_A_week8.json` by `python eval/run_eval.py --part A --golden eval/golden.jsonl --output eval/results_A_week8.json`. The 12 mistakes are look-alike breeds (English Foxhound vs Treeing Walker Coonhound, Toy vs Miniature Poodle, Siberian Husky vs Eskimo Dog); 11 of the 12 still have the right breed in the top 3. One Australian Kelpie is confidently called a Dingo.
- `identify_breed` turns phone photos upright from their EXIF orientation tag before the model sees them, and checks the `image_id` before loading the model.

```bash
python scripts/calibrate_classifier.py --fit-temperature   # writes temperature.json
python scripts/calibrate_classifier.py --sweep             # tables only, reuses the saved temperature
python scripts/calibrate_classifier.py --confidence-threshold 0.82 --top2-threshold 0.40 --reason "..."
```

## Get the trained model (no training needed)

```bash
python scripts/download_classifier.py
```

- Downloads `convnext_tiny_stanford120.pt` from the GitHub Release `breed-classifier-v1` into this folder and checks its SHA-256. A wrong or failed download is deleted and never replaces a working file.
- Run it again after pulling if `identify_breed` or the tests say the checkpoint is missing; it does nothing when the file is already correct.
- A copy from before the #3 class-name rename has a different hash and is replaced.

## Files in this folder

- `convnext_tiny_stanford120.pt` - model weights plus `class_names`, `num_classes`, `stanford_labels`. Not in git (112 MB, over the GitHub file limit): it is an asset of the GitHub Release `breed-classifier-v1`. Download it with the command below.
- `class_names.json` - the 120 breed names (`breeds_map.json` display names), in the order of the model outputs.
- `temperature.json`, `threshold.json` - calibration from issue #10.
- `metrics.json` - test accuracy, validation accuracy, settings and the accuracy after every epoch.

## Data

- Stanford Dogs, official split: 12,000 train images (100 per breed) and 8,580 test images.
- The official server (vision.stanford.edu) was down in November 2026, so `scripts/download_stanford_dogs.py` uses a Hugging Face mirror with the same split. The files are pinned to one revision and checked with SHA-256.
- 20% of the train images (2,400, 20 per breed) are held out as a validation split, listed in `data/splits/stanford_dogs_val.json`. The model never trains on them; issue #10 uses them for temperature scaling.
- The test split is never used to train or tune anything. It gives the accuracy in `metrics.json`, and the thresholds chosen on the validation split were afterwards checked on it (checked, not tuned). Golden Part A is taken from the test split, so its photos are never used for training either.

## Run it locally

Python 3.11. On an NVIDIA RTX 50-series GPU, PyTorch must be a CUDA 12.8 build or newer:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series only
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/download_stanford_dogs.py
.venv/Scripts/python scripts/train_classifier.py --model convnext_tiny --data-dir data/raw/stanford_dogs --epochs 20 --batch-size 32 --output-dir models/breed_classifier
.venv/Scripts/python -m pytest tests/test_classifier_checkpoint.py
```

- Without a GPU the script still runs (CPU or Apple `mps`), but it is much slower.

## Training settings

- Head replaced by `Linear(768, 120)`; all layers are fine-tuned.
- Augmentation: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `ColorJitter`; ImageNet mean and std.
- AdamW, learning rate 1e-4, weight decay 0.05, one epoch of warm-up then cosine decay, label smoothing 0.1, batch size 32, 20 epochs, mixed precision on GPU.
- The epoch with the best validation top-1 accuracy is kept.

## Known limits

- Stanford Dogs was built from ImageNet, and the model was pre-trained on ImageNet, so it may have seen some test images. The test accuracy can look better than it is; issue #20 adds photos from other sources.
- The class names were renamed to the `breeds_map.json` display names after #3 merged (46 of 120 changed, e.g. `Japanese Spaniel` -> `Japanese Chin`). Only the stored names changed, not the weights; a copy of the `.pt` from before the rename needs replacing.
- Three of the 120 classes are not dog breeds (Dingo, Dhole, African Wild Dog): `identify_breed` returns `kind: "wild_canid"` for them.
