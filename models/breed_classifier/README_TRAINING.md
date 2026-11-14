# Breed classifier: how to reproduce training (issue #6)

ConvNeXt-Tiny (ImageNet weights from torchvision) fine-tuned on Stanford Dogs, 120 breeds.

## Results

- Test split (8,580 images): top-1 accuracy 90.6%, top-3 accuracy 98.5%. Target for Week 7 was 70% and 85%.
- Validation split (2,400 images): top-1 90.4%, top-3 98.1%.
- Best epoch: 13 of 20. Accuracy was already 75% after the first epoch and stopped improving after about epoch 12.
- Training time: 16 minutes on an NVIDIA RTX 5050 Laptop GPU (8 GB), PyTorch 2.11 with CUDA 12.8.

## Files in this folder

- `convnext_tiny_stanford120.pt` - model weights plus `class_names`, `num_classes`, `stanford_labels`. Not in git (about 110 MB, over the GitHub file limit).
- `class_names.json` - the 120 breed names, in the order of the model outputs.
- `metrics.json` - test accuracy, validation accuracy, settings and the accuracy after every epoch.

## Data

- Stanford Dogs, official split: 12,000 train images (100 per breed) and 8,580 test images.
- The official server (vision.stanford.edu) was down in November 2026, so `scripts/download_stanford_dogs.py` uses a Hugging Face mirror with the same split. The files are pinned to one revision and checked with SHA-256.
- 20% of the train images (2,400, 20 per breed) are held out as a validation split, listed in `data/splits/stanford_dogs_val.json`. The model never trains on them; issue #10 uses them for temperature scaling.
- The test split is used once, at the end, for `metrics.json`. Golden Part A is taken from the test split, so its photos are never used for training either.

## Run it locally

Python 3.11. On an NVIDIA RTX 50-series GPU, PyTorch must be a CUDA 12.8 build or newer:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install pyarrow pytest
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
- Until `data/breeds_map.json` exists (issue #3), `class_names` are readable versions of the Stanford folder names (for example `German Short-haired Pointer`). After #3 is merged, re-run training or rename the classes so they match the `display` names in `breeds_map.json`.
- Three of the 120 classes are not dog breeds: dingo, dhole and African hunting dog.
