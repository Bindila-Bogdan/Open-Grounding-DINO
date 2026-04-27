# Open GroundingDINO RichArt Fine-Tuning

This is a fork of [Open GroundingDINO](https://github.com/longzw1997/Open-GroundingDino) adapted for fine-tuning on the [RichArt dataset](https://huggingface.co/datasets/MihaiBogdanBindila/RichArt), as part of the paper [Fine-Grained Cross-Modal Retrieval in Art via Region-Level Grounding of Symbolic Narratives](https://github.com/Bindila-Bogdan/Fine-Grained-Cross-Modal-Retrieval-in-Art).

This repository focuses exclusively on the fine-tuning pipeline; the MARGE-GD model extensions live in the main paper repository.

## Changes from upstream

- **Evaluation replaced** — upstream COCO evaluation removed entirely; [`util/evaluate.py`](util/evaluate.py) implements mAP@50 and mAP@50-95 against RichArt-style JSONL annotations. It is called as a subprocess after each training epoch and also supports direct invocation.
- **Training loop** — [`main.py`](main.py) reads per-epoch evaluation results from `intermediate_evaluation.json` for best-checkpoint tracking; epoch logs are written to a cumulative `logs.json` instead of the upstream `log.txt`.
- **COCO/OD infrastructure removed** — all COCO dataset loaders, panoptic eval, tools (`coco2odvg.py`, `inference_on_a_image.py`, etc.), slurm scripts, and COCO configs are removed; the fork is VG-only.
- **Dataset loader** — [`datasets/odvg.py`](datasets/odvg.py) is unchanged in behaviour; example RichArt-format annotation files are provided in [`data/train/`](data/train/) and [`data/val/`](data/val/).
- **Config** — [`config/cfg_odvg.py`](config/cfg_odvg.py) holds training hyperparameters. [`train_dist.sh`](train_dist.sh) and [`test_dist.sh`](test_dist.sh) hard-code `./weights/groundingdino_swint.pth` and `bert-base-uncased`.
- **PyTorch 2.x compatibility** — `engine.py` moved from the repo root to [`util/engine.py`](util/engine.py); `torch.cuda.amp.autocast` replaced with `torch.amp.autocast("cuda", ...)` in both [`util/engine.py`](util/engine.py) and [`models/GroundingDINO/transformer.py`](models/GroundingDINO/transformer.py); `use_reentrant=False` added to `checkpoint.checkpoint()` calls in the transformer.
- **Dependencies** — [`requirements.txt`](requirements.txt) removes `torch`/`torchvision`/`jsonlines` and adds `torchmetrics`.
- **Embedding extraction** — the fine-tuned weights produced here are loaded by the sibling `GroundingDINO/` codebase (see [Directory layout](#directory-layout)), which has been extended with extraction logic: visual embeddings from the final Cross-modality Decoder layer and textual embeddings from the Feature Enhancer, followed by average pooling and L2 normalization.
- **Note** — [`training_example.ipynb`](training_example.ipynb) is an unmodified upstream notebook using the Aquarium dataset; it was used only for testing purposes.

## Directory layout

[`setup.sh`](setup.sh) expects the following sibling directory layout:

```
parent/
├── GroundingDINO/        # IDEA-Research GroundingDINO, extended with embedding extraction code
└── Open-Grounding-DINO/  # this repository
```

## Setup

```bash
bash setup.sh
```

The script will:
1. Build and install GroundingDINO from `../GroundingDINO/`.
2. Create an `open_grounding_dino` conda environment (Python 3.12.3). Note: `conda activate` inside a shell script requires `conda init` to have been run in the shell first.
3. Install dependencies from [`requirements.txt`](requirements.txt).
4. Build and test the deformable attention extension in [`models/GroundingDINO/ops/`](models/GroundingDINO/ops/).

## Usage

### Fine-tuning

```bash
bash train_dist.sh <NUM_GPUS> <CONFIG> <DATASETS_JSON> <OUTPUT_DIR>

# example
bash train_dist.sh 1 config/cfg_odvg.py config/datasets_vg_test.json ./output
```

Expects a pretrained checkpoint at `./weights/groundingdino_swint.pth`.

### Evaluation

```bash
bash test_dist.sh <NUM_GPUS> <CONFIG> <DATASETS_JSON> <OUTPUT_DIR>
```
