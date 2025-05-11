# NeuroVR Pipeline

This repository contains the full multimodal deep learning pipeline for **NeuroVR**, an AI-powered diagnostic system for detecting Neuro-AIDS and related neurological disorders from MRI and EEG data. The system includes data preprocessing, model training, evaluation, interpretability with Grad-CAM, and experiment tracking with MLflow.

---

## 🧠 Features

- MRI and EEG preprocessing with data validation and normalization
- 3D CNN for MRI-based classification of neurodegenerative conditions
- LSTM-RNN model for EEG-based seizure/encephalopathy detection
- Fusion model combining MRI + EEG representations
- Automated data augmentation for MRI using `ImageDataGenerator`
- Hyperparameter tuning using `ParameterGrid` + K-Fold cross-validation
- Grad-CAM visualization for MRI interpretability
- Experiment tracking and model logging via MLflow
- Logging system with real-time file + console output

---



## ✅ Supported Datasets

- **MRI**: ADNI, HCP, fastMRI, OpenNeuro, FigShare HIV datasets
- **EEG**: CHB-MIT, PhysioNet, OpenNeuro EEG, seizure datasets from Kaggle

---

## 📌 Notes

- Ensure `.nii` and `.edf` files are correctly labeled (`hiv`, `seizure`) for automatic target assignment.
- Fusion model training requires equal sample size in EEG and MRI datasets.
- MRI Grad-CAM requires model layer named `last_conv`.

---

## 📄 License

© 2025 NeuroVR Research Team. All rights reserved.
Licensed under MIT License.

---

## 👩‍🔬 Citation

If you use this project for research, please cite the corresponding NeuroVR paper submitted to NeurIPS 2025.
