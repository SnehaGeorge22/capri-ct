# CapriCT: Causal-Aware SNR Prediction in CT Phantom Imaging

This repository contains models and code for predicting **Signal-to-Noise Ratio (SNR)** from CT phantom images using a **causal representation learning approach**. The models are based on **Variational Autoencoders (VAEs)** and support **ablation testing** to evaluate the contribution of various acquisition metadata: `voltage`, `time`, and `contrast agent`.

---

## 📌 Project Objectives

- Build a causal-aware SNR prediction model from CT phantom image data.
- Incorporate acquisition parameters (voltage, time, contrast agent) as structured metadata.
- Evaluate the influence of each variable through ablation and intervention.
- Simulate `do()` operations to interpret causal effects on SNR.
