# Detection of Multi-Modal Deepfakes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)

**A Systematic Comparison of Architectures for Audio-Visual Video-Level Classification**

Bachelor thesis by **Ali Shariati Najafabadi**, Department of Computer Science, TU Darmstadt, in cooperation with Fraunhofer SIT.
Reviewers: Prof. Dr. Martin Steinebach, Raphael Antonius Frick.

> 📄 **Full thesis:** [`docs/Bachelor_Thesis_Ali_Shariati_Najafabadi.pdf`](docs/Bachelor_Thesis_Ali_Shariati_Najafabadi.pdf)
> This README is only a short overview. Motivation, related work, design details, all results and the discussion are in the thesis.

## Overview

Many deepfake detectors look only at the video and give a binary real/fake answer. This project detects deepfakes at the **video level using both audio and video**, and compares three architectures under identical training and evaluation conditions.

Two formulations are supported:

- **Binary:** Real vs. Fake
- **Four-class:** Real, Fake Audio / Real Visual (FARV), Real Audio / Fake Visual (RAFV), Fake Audio / Fake Visual (FAFV). This shows *which* modality was manipulated.

## Architectures

All three phases share the same audio branch (HuBERT), preprocessing, training procedure and loss weights. Only the video encoder and the fusion strategy change.

| Phase | Video encoder | Audio encoder | Fusion | Params (trainable) |
|-------|---------------|---------------|--------|--------------------|
| 1 | ResNet-50 + BiLSTM | HuBERT | Concatenation | 124.1 M (57.8 M) |
| 2 | X3D-M | HuBERT | Concatenation | 97.5 M (31.4 M) |
| 3 | TimeSformer | HuBERT | Cross-attention | 217.3 M (104.5 M) |

Total loss = 0.6 · fusion + 0.2 · video + 0.2 · audio (auxiliary losses on each branch).

## Datasets

Trained and evaluated on [AV-Deepfake1M++](https://arxiv.org/abs/2311.15308), [FakeAVCeleb](https://arxiv.org/abs/2108.05080) and [DeepSpeak v2.0](https://arxiv.org/abs/2408.05366), including all cross-dataset combinations. **The datasets are not included in this repository.** Please obtain them from their original sources and respect their usage terms.

## Key Results (AUC)

| Setting | Phase 1 | Phase 2 | Phase 3 |
|---------|:------:|:------:|:------:|
| AV-Deepfake1M++ binary, in-domain | **0.968** | 0.948 | 0.769 |
| AV-Deepfake1M++ → FakeAVCeleb (binary) | 0.823 | **0.877** | 0.657 |
| AV-Deepfake1M++ → DeepSpeak (binary) | 0.453 | 0.460 | 0.536 |

Phase 3 uses concatenation fusion in the binary AV-Deepfake1M++ setting, because it outperformed cross-attention there (see thesis, Section 5.6.2).

**Main findings**

- The pipeline works well **in-domain**, but **cross-dataset generalization is limited**.
- The **audio branch generalizes better** than the video branch. The video branch often drops to chance level on unseen datasets.
- With uniform frame sampling the video branch overfits to dataset-specific artifacts. **Segment-based training** raises its cross-dataset AUC from 0.511 to 0.869 (Phase 1, four-class, AV-Deepfake1M++ → FakeAVCeleb).
- Fusion can be hurt by a poorly generalizing video branch, and **more complex architectures did not perform better**: Phase 3 is behind the simpler phases both in-domain and cross-dataset.

## Repository Structure

```
configs/        YAML configs for every phase / dataset / task
data/           Preprocessing (face crops, frames, audio caching) and dataloaders
models/         Video encoders, HuBERT audio branch, fusion modules, phase models
training/       Training loop, losses, metrics, checkpointing
evaluation/     Cross-dataset / per-modality evaluation, ROC, t-SNE, Grad-CAM
scripts/        Plotting helpers (confusion matrices, heatmaps, training curves, ...)
docs/           Bachelor thesis (PDF)
```

## Setup

The experiments used Python 3.10, PyTorch 2.5.1 and CUDA 12.1 on a single NVIDIA H100 (80 GB).

```bash
conda env create -f environment.yml
conda activate deepfake-detection
```

or, without conda:

```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess a dataset** into a sharded cache (16 frames at 224 px, 4 s of 16 kHz audio):

   ```bash
   python data/preprocessing/preprocess_deepfake.py --dataset av_deepfake --mode binary --cache-dir <CACHE_DIR>
   ```

2. **Set the dataset cache paths** in the config you want to use (`configs/`). The paths in the provided configs point to the original training server.

3. **Train:**

   ```bash
   python training/train.py --config configs/phase1_avdeepfake_50k_binary.yaml
   ```

4. **Evaluate** a checkpoint (also across datasets):

   ```bash
   python evaluation/evaluate_modality.py --config <CONFIG> --checkpoint <CKPT> --cache-dir <CACHE_DIR> --dataset <av_deepfake|fakeavceleb|deepspeak>
   ```

Trained model weights are not included.

## License

The code is released under the [MIT License](LICENSE). The datasets and pretrained models used (e.g. `facebook/hubert-base-ls960`) have their own licenses and terms of use.

## Ethical Note

Deepfake detection is a dual-use topic. A detector should support, not replace, an overall forensic assessment. The datasets contain faces and voices of real people, so their license and data-protection requirements apply. See Section 6.3 of the thesis.

## Citation

```bibtex
@thesis{shariati2026multimodal,
  author = {Shariati Najafabadi, Ali},
  title  = {Detection of Multi-Modal Deepfakes: A Systematic Comparison of Architectures for Audio-Visual Video-Level Classification},
  school = {Technische Universit{\"a}t Darmstadt},
  type   = {Bachelor's thesis},
  year   = {2026}
}
```
