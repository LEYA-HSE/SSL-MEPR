# SSL-MEPR: A Semi-Supervised Multi-Task Cross-Domain Learning Framework for Multimodal Emotion and Personality Recognition

# Facial Emotion and Personality Prediction

This branch includes code for:

* Extracting face embeddings from raw videos
* Training independent models on CMU-MOSEI and First Impressions V2 datasets
* A multitask fusion model that jointly processes both modalities
* End-to-end inference from raw `.mp4` video

  Weights for all used models can be accessed via [this](https://drive.google.com/drive/folders/1BsJ8F_vM-SHG-IHVTYutAjC21nkimNTd?usp=sharing) link.

---

## File Overview

| File                          | Purpose                                                  |
| ----------------------------- |----------------------------------------------------------|
| `get_cmu_mosei_embeddings.py` | Extract face embeddings from CMU-MOSEI videos            |
| `get_fiv2_embeddings.py`      | Extract face embeddings from First Impressions V2 videos |
| `emonext_model.py`            | Defines the `EmoNeXt` ConvNeXt-based embedding extractor |
| `cmu_mosei_model.ipynb`       | Trains Transformer on CMU-MOSEI emotion classification   |
| `fiv2_mosei_model.ipynb`      | Trains Transformer on FIV2 personality regression        |
| `fusion_model.ipynb`          | Trains fusion model with cross-attention + frozen heads  |
| `total_inference.ipynb`       | Complete inference pipeline: video → traits + emotions   |

---

## Models Used

### `EmoNeXt` (in `emonext_model.py`)

* A ConvNeXt variant with STN preprocessor and self-attention.
* Extracts 30 × 1024 embedding matrix per video.
* Used by all embedding pipelines.

### Transformer classifier and regressor (in notebooks)

* Independent Transformers trained for CMU-MOSEI and FIV2 using 30×1024 embeddings.
* Output: softmax emotion distribution (7D) or 5 personality traits (OCEAN).

### `MultiTaskFusionModel` (in `fusion_model.ipynb`)

* Uses frozen pretrained emotion and trait models as feature extractors.
* Two Transformer encoders + cross-attention.
* Outputs both traits and emotions in a single pass.

---

## Embedding Extraction Scripts

### 1. `get_cmu_mosei_embeddings.py`

Extracts 30 face embeddings per video (as `.pt` files) for CMU-MOSEI.

**Usage:**

```bash
python get_cmu_mosei_embeddings.py \
    --split train \
    --base_path emo_video \
    --output_root emo_video_embeddings
```

**Arguments:**

* `--split`: One of `train`, `dev`, `test`
* `--base_path`: Root folder with videos (default: `emo_video`)
* `--output_root`: Where to store `.pt` embeddings

**Output:**

* Saves files like `emo_video_embeddings/train/video123.pt`
  Each file contains a dict: `{ "emb": Tensor[30, 1024], "length": 30 }`

---

### 2. `get_fiv2_embeddings.py`

Same as above, but for First Impressions V2. Supports multiple folders (since original dataset has this weird split into 75 subdirectories).

**Usage:**

```bash
python get_fiv2_embeddings.py \
    --split train \
    --folders training80_01 training80_02 \
    --base_path FirstImpressionsV2 \
    --output_root FirstImpressionsV2_embeddings
```

**Arguments:**

* `--split`: One of `train`, `val`, `test`
* `--folders`: List of folder names to process (e.g. `training80_01`)
* `--base_path`: Root dataset folder
* `--output_root`: Output root directory

**Output:**

* Saves files like `FirstImpressionsV2_embeddings/train/video123.pt`

---

## Training Scripts (.ipynb files)

### `cmu_mosei_model.ipynb`

* Loads CMU-MOSEI emotion labels and precomputed embeddings (30×1024 sequences)
* Trains a Transformer on embeddings
* Loss: balanced `KLDivLoss` with softmax target distributions
* Outputs: `cmu_mosei_best_checkpoint.pth`

---

### `fiv2_mosei_model.ipynb`

* Loads FIV2 labels (OCEAN only, no `interview`) and embeddings
* Trains a Transformer on same 30×1024 format
* Loss: `MSELoss` + metrics like CCC and MAE
* Outputs: `fiv2_best_checkpoint.pth`

---

### `fusion_model.ipynb`

* Loads both pretrained models
* Freezes their parameters
* Feeds their outputs into cross-attention fusion architecture
* Shared MLP + dual heads for:

  * 5-dim regression (traits)
  * 7-dim softmax (emotions)
* Saved as: `multitask_fusion_model.pth`

---

## Inference

### `total_inference.ipynb`

Given a raw `.mp4` video:

1. Runs face detection and frame sampling
2. Extracts 30×1024 embedding via `EmoNeXt`
3. Loads `MultiTaskFusionModel` with pretrained frozen heads
4. Returns:

   * Emotion distribution: softmax over 7 classes
   * Personality traits: 5 continuous OCEAN scores


---

## Dependencies

View `requirements.txt` file for details.
