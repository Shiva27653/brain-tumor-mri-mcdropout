
```markdown
# Brain Tumor MRI Classification with MC Dropout Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning pipeline for brain tumor classification from MRI scans using
**EfficientNetB0** with **Monte Carlo Dropout** for predictive uncertainty estimation.
The model classifies tumors into 4 categories while quantifying confidence per prediction —
enabling selective prediction: reject uncertain cases to achieve 94%+ accuracy on retained samples.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy (deterministic) | 68.78% |
| Test Accuracy (MC Dropout, 50 passes) | **70.30%** |
| Macro F1 Score | 0.6750 |
| Expected Calibration Error (ECE) | 0.1153 |

### Uncertainty-Controlled Accuracy (Abstention Curve)

By rejecting samples where predictive entropy exceeds a threshold:

| Coverage (samples retained) | Accuracy |
|---|---|
| 10% | 100.00% |
| 20% | 98.73% |
| 30% | 94.07% |
| 50% | 88.83% |
| 100% (no rejection) | 70.30% |

> Rejecting the 30% most uncertain predictions yields **94% accuracy** on retained samples.

---

## Visualizations

| Plot | Description |
|---|---|
| `results/uncertainty_per_class.png` | Most uncertain samples per class |
| `results/entropy_histogram.png` | Entropy distribution: correct vs incorrect predictions |
| `results/calibration_curve.png` | Reliability diagram + ECE |
| `results/gradcam_per_class.png` | Grad-CAM heatmaps per class |
| `results/abstention_curve.png` | Accuracy vs coverage under uncertainty rejection |

---

## Model Architecture

```
EfficientNetB0 (ImageNet pretrained, frozen base)
       ↓
GlobalAveragePooling2D
       ↓
Dense(256, ReLU)
       ↓
BatchNormalization
       ↓
MCDropout(0.4)       ← stays active at inference
       ↓
Dense(4, Softmax)
```

**Training Strategy:**
- Phase A: Frozen base, head only — Adam lr=1e-3, 20 epochs
- Phase B: Last 40 layers unfrozen — Adam lr=5e-6, 25 epochs
- CLAHE preprocessing, class-weighted loss, EarlyStopping

---

## MC Dropout Inference

Standard dropout is disabled at test time. MCDropout overrides this:

```python
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
```

At inference, 50 stochastic forward passes are run per image.
Predictive entropy is computed from the mean probability distribution:

**H = −Σ p · log(p + ε)**

High entropy = high uncertainty = model should abstain.

---

## Dataset

[Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) — Kaggle

| Class | Train | Test |
|---|---|---|
| Glioma Tumor | 826 | 100 |
| Meningioma Tumor | 822 | 115 |
| No Tumor | 395 | 105 |
| Pituitary Tumor | 827 | 74 |

---

## Repository Structure

```
brain-tumor-mri-mcdropout/
├── results/
│   ├── uncertainty_per_class.png
│   ├── entropy_histogram.png
│   ├── calibration_curve.png
│   ├── gradcam_per_class.png
│   └── abstention_curve.png
├── brain_tumor_best.keras
├── train.py                  ← coming soon
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Shiva27653/brain-tumor-mri-mcdropout.git
cd brain-tumor-mri-mcdropout
pip install -r requirements.txt
python train.py
```

---

## License

MIT
```
