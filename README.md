# Contrastive Predictive Coding for Audio and Images  
### Multi-Modal Context Learning with Hybrid Negative Sampling

Course: **EE782 – Deep Learning for Vision**  
Instructor: **Prof. Amit Sethi**  
Authors: **Anirudh Garg, Satyankar Chandra**

---

## 1. Overview

This project implements a **multi-modal Contrastive Predictive Coding (CPC)** framework that works on:

- **Audio** (synthetic sine waves with frequency-based labels)  
- **Images** (two-class subset of CIFAR-10: airplane vs automobile)  
- (Optionally) **Synthetic multi-modal pairs** (audio + images)

We extend the classic CPC setup in three main ways:

1. **Multi-modal architecture**  
   - Separate encoders for audio and images  
   - **Shared GRU-based context model** that summarizes sequences of latents from any modality

2. **Hybrid negative sampling**  
   - In-batch negatives  
   - **MoCo-style queue** of past latents  
   - **Synthetic “hard” negatives** (Gaussian-perturbed latents + time/position shuffled latents)  
   - Optional **cross-modal negatives** when multiple modalities are present in the batch

3. **Rich representation evaluation**  
   - **Alignment & uniformity** metrics for the context space  
   - **Linear probe** (logistic regression) on frozen contexts  
   - **t-SNE visualizations** of learned context embeddings

The project is CPU-friendly: audio is synthetic and cheap to generate; images are restricted to 2 CIFAR-10 classes (~2k images/class) to keep training time reasonable while still being non-trivial.

---

## 2. Repo Structure

A typical layout (your exact paths may differ slightly):

```text
.
├── cpc_project.py          # Main training & evaluation script (CLI)
├── prepare_cifar10.py      # Script to download + extract 2-class CIFAR-10
├── requirements.txt        # (Optional) Python dependencies
├── data/
│   ├── cifar10_raw/        # Raw CIFAR-10 (downloaded by torchvision)
│   └── images/
│       ├── class0/         # Airplanes
│       └── class1/         # Automobiles
├── outputs_audio/          # Results for audio experiments
│   ├── training_curve.png
│   └── tsne_audio.png
├── outputs_image/          # Results for image experiments
│   ├── training_curve.png
│   └── tsne_image.png
├── outputs_multimodal/     # (Optional) results for multimodal runs
│   ├── training_curve.png
│   └── tsne_audio.png
└── report/
    └── cpc_report.tex      # LaTeX report (arXiv-style)
````

---

## 3. Installation

### 3.1. Create and activate virtualenv (recommended)

```bash
cd ~/wd/rough   # or wherever you cloned the repo

python3 -m venv venv
source venv/bin/activate
```

### 3.2. Install dependencies

Minimal set (adjust versions to your environment if needed):

```bash
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib
```

If you want to be explicit, you can create a `requirements.txt` like:

```text
torch
torchvision
torchaudio
scikit-learn
matplotlib
```

and then run:

```bash
pip install -r requirements.txt
```

> **Note:** Audio uses `torchaudio.transforms.MelSpectrogram`.
> If torchaudio complains about FFmpeg/TorchCodec in your environment, you can still run **audio in mock/synthetic mode** because waveforms are generated in pure PyTorch.

---

## 4. Data Preparation

### 4.1. CIFAR-10 subset (images)

Run:

```bash
python prepare_cifar10.py
```

This will:

* Download CIFAR-10 (train set) into `data/cifar10_raw/`
* Extract **only** class `0` (airplane) and class `1` (automobile)
* Save up to `max_per_class` images for each, by default 2000:

Folder structure:

```text
data/images/
    class0/ 00000.png, 00001.png, ...
    class1/ 00000.png, 00001.png, ...
```

These are used by `cpc_project.py` when running with `--modality image --use-mock-data false`.

### 4.2. Audio data

For this project, **no disk audio dataset is required** for the main experiments:

* Audio waveforms are **synthetic sine waves** generated on-the-fly in `MockAudioDataset`.
* Each sample: single sine wave with random frequency in `[150, 900]` Hz + Gaussian noise.
* Label: `0` for low frequency (`< 500 Hz`), `1` for high frequency (`>= 500 Hz`).

So you don’t need to download LJ-Speech or any other audio corpus to reproduce the main results.

---

## 5. Running Experiments

All main experiments are run via **`cpc_project.py`**, which takes command line flags to select modality, data source, and output directory.

### 5.1. CLI arguments (summary)

The script supports (names may differ slightly depending on your exact version, but these are the ones used in the project):

* `--modality {audio,image,multimodal}`
* `--use-mock-data {true,false}`
* `--image-root PATH` (required when `use-mock-data=false` & modality uses images)
* `--audio-root PATH` (if you add real audio data later)
* `--out-dir PATH`
* `--epochs N`
* `--neg-mode {inbatch,queue,queue+synthetic}`
* `--batch-size N` (if included)
* `--z-dim`, `--c-dim`, `--queue-size`, etc. (depending on your version)

You can always inspect available flags via:

```bash
python cpc_project.py --help
```

---

### 5.2. Audio-only CPC (synthetic)

```bash
python cpc_project.py \
  --modality audio \
  --use-mock-data true \
  --out-dir ./outputs_audio \
  --epochs 10 \
  --neg-mode queue+synthetic
```

This will:

* Train CPC on synthetic sine-wave audio
* Use:

  * **Audio encoder** → Mel-spectrogram + small CNN
  * **Shared GRU context** (dimension 256)
  * **Hybrid negative sampler** (queue + synthetic negatives)
* Save:

  * InfoNCE training curve: `outputs_audio/training_curve.png`
  * t-SNE plot of contexts: `outputs_audio/tsne_audio.png`
  * Model weights: `outputs_audio/cpc_model.pth`
* Print:

  * Alignment & uniformity metrics
  * Linear probe accuracy on audio contexts (usually ≈ 99%)

---

### 5.3. Image-only CPC (CIFAR-10 subset)

First, ensure the CIFAR images are prepared:

```bash
python prepare_cifar10.py
```

Then run:

```bash
python cpc_project.py \
  --modality image \
  --use-mock-data false \
  --image-root ./data/images \
  --out-dir ./outputs_image \
  --epochs 20 \
  --neg-mode queue+synthetic
```

This will:

* Train CPC on two-class CIFAR-10 (airplane vs automobile)
* Use the same **GRU context model** and **negative sampler** as for audio, but with an **image encoder**
* Save:

  * `outputs_image/training_curve.png`
  * `outputs_image/tsne_image.png`
  * `outputs_image/cpc_model.pth`
* Print:

  * Alignment & uniformity metrics for image contexts
  * Linear probe accuracy (moderate; weaker than audio)

---

### 5.4. Multi-modal synthetic CPC (optional)

If you enable the **Mock multi-modal dataset** in `cpc_project.py` (using `MockMultiModalDataset`), you can run:

```bash
python cpc_project.py \
  --modality multimodal \
  --use-mock-data true \
  --out-dir ./outputs_multimodal \
  --epochs 10 \
  --neg-mode queue+synthetic
```

This will:

* Train CPC on paired `(audio, image)` synthetic data
* Optionally use **cross-modal negatives**: audio latents as negatives for images and vice versa
* Save:

  * `outputs_multimodal/training_curve.png`
  * `outputs_multimodal/tsne_audio.png` (and/or `tsne_image.png` depending on what you log)

---

## 6. Running in the Background

On a remote machine / server, you can run longer experiments in the background with:

```bash
# Example: image run in background, logging to a file
nohup python cpc_project.py \
  --modality image \
  --use-mock-data false \
  --image-root ./data/images \
  --out-dir ./outputs_image \
  --epochs 20 \
  --neg-mode queue+synthetic \
  > logs_image.txt 2>&1 &
```

Useful commands:

```bash
# List running CPC jobs
ps aux | grep cpc_project.py

# Tail logs
tail -f logs_image.txt
```

---

## 7. Reproducing Report Figures

The LaTeX report (`report/cpc_report.tex`) expects:

* **Audio training curve:**
  `outputs_audio/training_curve.png`
* **Audio t-SNE:**
  `outputs_audio/tsne_audio.png`
* **Image training curve:**
  `outputs_image/training_curve.png`
* **Image t-SNE:**
  `outputs_image/tsne_image.png`
* (Optional multimodal figures:)
  `outputs_multimodal/training_curve.png`
  `outputs_multimodal/tsne_audio.png`

After running the corresponding experiments, compile the report:

```bash
cd report
pdflatex cpc_report.tex
pdflatex cpc_report.tex  # (twice for refs)
```

The final PDF can be uploaded to arXiv or submitted as the EE782 project report.

---

## 8. What This Project Shows (Conceptually)

* CPC **strongly captures predictive structure** in audio:

  * Contexts are nearly linearly separable by frequency
  * Alignment & uniformity are good
  * t-SNE yields two smooth, well-separated clusters
* The **same CPC setup struggles more on images**:

  * Training loss is noisier
  * Classes overlap substantially in context space
  * Metrics and t-SNE both confirm weaker geometry

This asymmetry is exactly the point:
negative sampling and architecture that work beautifully for audio do **not automatically** transfer to images. The project demonstrates this clearly and provides a clean, reproducible setup to study:

* Multi-modal context learning
* Hybrid negative sampling
* Different representation quality metrics

---

## 9. References

* A. v. d. Oord, Y. Li, O. Vinyals,
  **Representation Learning with Contrastive Predictive Coding**, arXiv:1807.03748
* N. Saunshi et al.,
  **A Theoretical Analysis of Contrastive Unsupervised Representation Learning**, arXiv:1902.09229
* J. Robinson et al.,
  **On the Role of Negative Samples in Contrastive Learning**, ICLR 2022
* F. Wang, H. Liu,
  **Understanding the Behaviour of Contrastive Loss**, CVPR 2021