# 🧠 Traffic Sign Classification  
### A Two-Phase Investigation of Resolution, Architecture, and Dataset Limitations  

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Framework](https://img.shields.io/badge/framework-TensorFlow%2FKeras-orange.svg)]()
[![Paper](https://img.shields.io/badge/Paper-Research%20Report-blueviolet)]()
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB-red.svg)]()

---

**Author:** Ahmed Ibrahim Muhammed  
**Location:** Cairo, Egypt  
**Date:** November 7, 2025  
**Repository:** [https://github.com/AhmadElsisy](https://github.com/AhmadElsisy)  

---

## 🧭 Abstract

This repository accompanies the research titled  
**“Traffic Sign Classification: A Two-Phase Investigation of Resolution, Architecture, and Dataset Limitations.”**  

The study explores why models achieving >98% benchmark accuracy often collapse in real-world deployment.  
Two controlled experimental phases isolate the effects of **input resolution** and **architectural sophistication (CNN vs Hybrid CNN-Transformer)** on model generalisation.  

> **Core finding:** Dataset characteristics fundamentally constrain real-world deployment success more than technical optimisation.

When trained on GTSRB (German Traffic Sign Recognition Benchmark), even sophisticated models with transformer attention achieve high test accuracy but poor real-world success, validating that benchmark performance is a poor proxy for deployment readiness.

---

## 📑 Keywords
`Traffic Sign Recognition` · `CNN` · `Vision Transformer` · `Resolution Scaling` · `Domain Shift` · `Benchmark Illusion` · `Data-Centric AI` · `Deployment Generalisation`

---

## 🧩 Research Overview

### 🎯 Phase 1 – Resolution vs. Generalisation
**Objective:** Determine whether input resolution improves real-world transferability.

- **Resolutions tested:** 64×64 → 128×128  
- **Architecture:** 4-block CNN  
- **Best performing setup:** 80×80, moderate augmentation  
- **Results:**
  - **Test Accuracy:** 98.7%  
  - **Real-World Success:** 9.1%

> Lower resolutions yielded stronger generalisation. More pixels led to overfitting dataset-specific artefacts.

---

### ⚙️ Phase 2 – Architectural Sophistication
**Objective:** Test whether adding global transformer attention overcomes Phase 1’s limitations.

- **Architecture:** Hybrid CNN + Transformer (3 attention blocks, 8 heads each)  
- **Result:**
  - **Test Accuracy:** 98.58%  
  - **Real-World Success:** 16.7%

> Transformer attention improved geometric and numeric sign recognition (e.g., Stop, 60 km/h) but regressed on pictorial signs due to background distraction.

---

## 📊 Summary of Findings

| Model | Benchmark Accuracy | Real-World Success | Observation |
|--------|-------------------:|-------------------:|--------------|
| CNN (80×80) | 98.7% | 9.1% | Benchmark illusion – fails in the wild |
| Hybrid CNN-Transformer | 98.58% | 16.7% | Marginal gain, still dataset-bound |
| Root Cause | — | — | Dataset domain gap (idealised vs real-world) |

---

## 🧠 Insights

### ❗ The Benchmark Illusion  
Both models exceeded **98%** test accuracy yet failed on over **75%** of real-world cases.  
Academic test sets do not reflect deployment variability — lighting, angles, clutter, or regional sign differences.

### 🌍 The Data-Centric Lesson  
Architecture helps **only if** training and deployment distributions align.  
Better data diversity yields higher real-world success than any model tweak.

> “A flawed dataset will blind even the most sophisticated model.”

---

## ⚙️ Technical Implementation

### 🧱 Phase 1 – Pure CNN
```python
Conv2D → BatchNorm → MaxPool → Dropout (×4)
GlobalAveragePooling → Dense(256, relu) → Dropout → Dense(43, softmax)
```
* Parameters: ~5M

* Optimiser: Adam (1e-4)

* Training Time: 40–75 min (RTX 2060)

* Mixed precision (FP16) enabled

### Phase 2 – Hybrid CNN + Transformer
```python
CNN backbone (512 filters)
→ reshape to (25 tokens)
→ 3 Transformer blocks (8 heads, SwiGLU activation)
→ GlobalAveragePooling → Dense(512) → Dropout → Dense(43, softmax)
```

* Parameters: ~85M

* Quantization-Aware Training for deployment efficiency

* Early stopped at epoch 36

* Training time: ~50 min


## 🔬 Future Research Directions
* Dataset Expansion: Incorporate Mapillary, LISA, and CTSD datasets for global diversity.

* Architectural Exploration: Compare DETR, Swin Transformer, and CoAtNet.

* Domain Adaptation: Explore self-supervised fine-tuning and few-shot learning.

* Deployment: Integrate detection + classification pipelines for natural scenes.

* Human-in-the-loop Validation: Enable feedback-based improvement for edge deployment.

## 🏗️ Practical Takeaways
1. Evaluate on deployment data early.

2. Invest in data quality before model complexity.

3. Benchmark accuracy ≠ production readiness.

4. Negative results are valuable — they reveal the limits of our assumptions.

## 📜 Citation
If you use this repository or reference this work, please cite:

Ahmed Ibrahim Muhammed.
Traffic Sign Classification: A Two-Phase Investigation of Resolution, Architecture, and Dataset Limitations.
Cairo, Egypt. July 2025.
[Git Hub Repo](https://github.com/AhmadElsisy)

```bibtex
@misc{muhammed2025traffic,
  title   = {Traffic Sign Classification: A Two-Phase Investigation of Resolution, Architecture, and Dataset Limitations},
  author  = {Ahmed Ibrahim Muhammed},
  year    = {2025},
  note    = {Independent Research, Cairo, Egypt},
  url     = {https://github.com/AhmadElsisy}
}
```
## 🙏 Acknowledgments
This research was conducted independently to challenge benchmark-driven complacency in computer vision.
It demonstrates that honest evaluation and negative findings are vital steps towards robust AI deployment.

“Benchmark success means little if the system fails in the field.”

## 📚 References
* Stallkamp, J. et al. (2012). Man vs. Computer: Benchmarking ML Algorithms for Traffic Sign Recognition. Neural Networks.

* Neuhold, G. et al. (2017). The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes. ICCV.

* Dosovitskiy, A. et al. (2021). An Image Is Worth 16×16 Words. ICLR.

* Vaswani, A. et al. (2017). Attention Is All You Need.

* Ganin, Y. & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. ICML.

* Tzeng, E. et al. (2017). Adversarial Discriminative Domain Adaptation. CVPR.