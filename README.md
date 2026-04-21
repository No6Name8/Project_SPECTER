```
███████╗██████╗ ███████╗ ██████╗████████╗███████╗██████╗
██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗
███████╗██████╔╝█████╗  ██║        ██║   █████╗  ██████╔╝
╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══╝  ██╔══██╗
███████║██║     ███████╗╚██████╗   ██║   ███████╗██║  ██║
╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
```

# Signal Processing & Extraction of Covert Transmissions in Electromagnetic Regimes

> **"Seeing what every other system calls nothing."**

---

## What is SPECTER?

SPECTER is a machine learning system built to detect and characterise covert radio transmissions buried in noise — the kind of signals that standard classification tools fail on entirely. It ingests raw IQ (in-phase/quadrature) radio samples, identifies the modulation type even when the signal-to-noise ratio is deeply negative, flags transmissions that don't match any known waveform as potential threats, scores each detection on a 0–100 threat scale, and estimates the transmitter's geographic position. The result is a single end-to-end pipeline that takes antenna data in and produces an actionable intelligence report out — something no off-the-shelf tool currently does in a unified way.

---

## The Problem

<table>
<tr>
<td align="center" width="33%">

**📉 SNR Collapse**

Standard classifiers achieve reasonable accuracy above 0 dB SNR. Below −6 dB — where covert transmitters deliberately operate — accuracy falls to near-random. SPECTER targets exactly this regime.

</td>
<td align="center" width="33%">

**❓ Closed-Set Blindness**

Every classifier trained on a fixed set of waveforms will confidently misclassify a waveform it has never seen. SPECTER's open-set detector flags novel signals rather than forcing them into a wrong category.

</td>
<td align="center" width="33%">

**🔗 Pipeline Gap**

Classify, detect, score, and locate are four separate problems in the literature. No unified pipeline exists that chains them. SPECTER does.

</td>
</tr>
</table>

---

## What SPECTER Does

### 🔍 Low-SNR Classifier — `models/specter_cnn.py`

A deep residual CNN with four staged ResBlocks, batch normalisation, and global average pooling. Trained with SNR-aware loss weighting that gives 3× gradient emphasis to the −20 dB to −6 dB regime where baseline models collapse. Outputs raw logits (no softmax) to keep the representation available for open-set analysis.

### ⚠️ Open-Set Detector — `models/open_set.py`

Energy Score detection (Liu et al., NeurIPS 2020):

```
E(x) = −log Σ exp(fₖ(x))
```

Lower energy means the model is confident the input is a known class. Higher energy means it's something the model has never seen. A threshold at the 95th percentile of known-class energies is used to reject novel transmissions. AUROC evaluation compares this against the softmax MSP baseline.

### 🎯 Threat Scorer — `scoring/threat_scorer.py`

A weighted scoring formula maps classifier outputs to a 0–100 threat score:

| Component | Weight | Logic |
|---|---|---|
| Uncertainty | 0.25 | `1 − confidence` |
| SNR Danger | 0.30 | Linear: −20 dB → 1.0, +30 dB → 0.0 |
| Unknown Penalty | 0.35 | Binary 1.0 if open-set rejected |
| Burst Suspicion | 0.10 | Bursts < 50 ms score highest |

Levels: **LOW** < 40 · **MEDIUM** 40–69 · **CRITICAL** ≥ 70

### 📍 TDOA Geolocation — `geolocation/tdoa_simulator.py`

Time Difference of Arrival simulation using a 3-receiver array. Least-squares hyperbolic intersection (Levenberg-Marquardt) estimates transmitter position and returns a confidence radius. Triggered automatically on MEDIUM and CRITICAL signals.

---

## Pipeline

```
┌───────────┐     ┌────────────┐     ┌──────────────┐     ┌─────────────┐
│  RAW IQ   │────▶│ PREPROCESS │────▶│   CLASSIFY   │────▶│  OPEN SET   │
│ (antenna) │     │ normalise  │     │ SpecterCNN   │     │   detect    │
└───────────┘     │ transpose  │     │ 24 waveforms │     │ energy score│
                  └────────────┘     └──────────────┘     └──────┬──────┘
                                                                  │
                  ┌───────────┐     ┌────────────┐               │
                  │   ALERT   │◀────│   LOCATE   │◀──────────────┤
                  │ dashboard │     │    TDOA    │     ┌──────────▼──────────┐
                  │  / log    │     │ simulation │     │        SCORE        │
                  └───────────┘     └────────────┘     │  0–100 threat level │
                                                        └─────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/specter.git
cd specter

# 2. Install dependencies
pip install -r specter/requirements.txt

# For GPU support (CUDA 12.6):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Download the dataset:**

RadioML 2018.01A is available on Kaggle:
👉 https://www.kaggle.com/datasets/pinxau1000/radioml2018

The dataset is ~19 GB and is **not included** in this repository.
After downloading, place the file at:

```
specter/data/raw/radio_ML/GOLD_XYZ_OSC.0001_1024.hdf5
```

```bash
# 3. Train both models (50 epochs, batch size 512)
cd specter
python train.py

# Train only SPECTER on low-SNR data:
python train.py --model specter --low_snr_only --epochs 30

# 4. Run full benchmark evaluation
python evaluate.py

# 5. Launch the app
streamlit run app.py
```

---

## Project Structure

```
Project_SPECTER/
├── specter/
│   ├── data/
│   │   ├── dataset.py              ← lazy HDF5 loader, DataLoader factory
│   │   └── raw/
│   │       └── radio_ML/
│   │           └── GOLD_XYZ_OSC.0001_1024.hdf5   ← NOT in repo (19 GB)
│   │
│   ├── models/
│   │   ├── baseline_cnn.py         ← VT-CNN2 (O'Shea et al. 2016)
│   │   ├── specter_cnn.py          ← deep residual CNN, SNR-aware loss
│   │   └── open_set.py             ← energy score + softmax baseline
│   │
│   ├── scoring/
│   │   └── threat_scorer.py        ← weighted 0–100 threat scoring
│   │
│   ├── geolocation/
│   │   └── tdoa_simulator.py       ← TDOA least-squares position estimator
│   │
│   ├── pipeline/
│   │   └── specter_pipeline.py     ← end-to-end SPECTERPipeline class
│   │
│   ├── results/                    ← generated by evaluate.py
│   │   ├── accuracy_vs_snr.csv
│   │   ├── accuracy_vs_snr.png
│   │   ├── confusion_baseline.png
│   │   ├── confusion_specter.png
│   │   ├── open_set_auroc.png
│   │   └── open_set_threshold.pkl  ← committed after training
│   │
│   ├── train.py                    ← CLI training script
│   ├── evaluate.py                 ← full benchmark evaluation
│   ├── app.py                      ← Streamlit dashboard
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## Benchmark Results

> Results will be updated after training completes on the full RadioML 2018.01A dataset.

| Metric | Baseline (VT-CNN2) | SPECTER-CNN |
|---|---|---|
| Overall Accuracy | TBD | TBD |
| Low-SNR Accuracy (−20 to −6 dB) | TBD | TBD |
| Open-Set AUROC | N/A | TBD |
| Inference Latency (CPU, batch=1) | TBD | TBD |

Per-SNR accuracy curves, confusion matrices, and the open-set ROC curve are generated automatically by `evaluate.py` and saved to `specter/results/`.

---

## Dataset

| Property | Value |
|---|---|
| Name | RadioML 2018.01A |
| Creator | DeepSig Inc. |
| Size | ~19 GB (HDF5) |
| Samples | 4,096,000 |
| Classes | 24 modulation types |
| SNR range | −20 dB to +30 dB (step 2 dB) |
| Frame length | 1024 IQ samples |
| Source | [Kaggle — pinxau1000/radioml2018](https://www.kaggle.com/datasets/pinxau1000/radioml2018) |

The dataset is **not included** in this repository due to file size.
The folder structure is preserved with a `.gitkeep` file at `specter/data/raw/`.

**Modulation classes (24):**
`OOK · 4ASK · 8ASK · BPSK · QPSK · 8PSK · 16PSK · 32PSK · 16APSK · 32APSK · 64APSK · 128APSK · 16QAM · 32QAM · 64QAM · 128QAM · 256QAM · AM-SSB-WC · AM-SSB-SC · AM-DSB-WC · AM-DSB-SC · FM · GMSK · OQPSK`

---

## Competition Alignment

| Criterion | How SPECTER Addresses It |
|---|---|
| **Signal Classification** | Deep residual CNN trained on all 24 RadioML 2018.01A classes with SNR-aware loss weighting |
| **Low-SNR Performance** | 3× loss weight for −20 to −6 dB samples; residual architecture preserves gradient signal at depth |
| **Open-Set Detection** | Energy Score (NeurIPS 2020) with 95th-percentile threshold; AUROC evaluation vs softmax baseline |
| **Threat Assessment** | Configurable weighted scorer combining uncertainty, SNR, open-set flag, and burst duration |
| **Geolocation** | TDOA least-squares estimator with 3-receiver array; confidence radius output; matplotlib visualisation |
| **Unified Pipeline** | Single `SPECTERPipeline` class: IQ in → `SPECTERResult` out, covering all five stages |
| **Reproducibility** | Seeded splits, checkpoint saving, CSV benchmark outputs, `requirements.txt` |
| **Baseline Comparison** | VT-CNN2 (O'Shea et al. 2016) re-implemented exactly; identical eval interface for direct comparison |

---

## References

- O'Shea, T., Corgan, J., Clancy, T.C. (2016). *Convolutional Radio Modulation Recognition Networks.* arXiv:1602.04105
- Liu, W., Wang, X., Owens, J., Li, Y. (2020). *Energy-based Out-of-distribution Detection.* NeurIPS 2020.
- Hendrycks, D., Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017.
- O'Shea, T., West, N. (2016). *Radio Machine Learning Dataset Generation with GNU Radio.* GNU Radio Conference.

---

<div align="center">

Built for competition. Defensive application only.

*SPECTER — Seeing what every other system calls nothing.*

</div>
