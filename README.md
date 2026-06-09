# 🎵 Music Instrument Recognition Using CRNN

## 📌 Project Overview

This project implements a **Convolutional Recurrent Neural Network (CRNN)** for automatic musical instrument recognition from audio recordings. The model combines the feature extraction capabilities of Convolutional Neural Networks (CNNs) with the temporal sequence modeling power of Bidirectional Gated Recurrent Units (BiGRUs).

The system processes audio signals by converting them into **Log-Mel Spectrograms**, enabling the model to learn both spectral and temporal characteristics of musical instruments.

---

## 🎯 Objectives

- Classify musical instruments from audio recordings.
- Learn meaningful spectral features using CNN layers.
- Capture temporal dependencies using Bidirectional GRU layers.
- Build a robust deep learning pipeline for audio classification.
- Evaluate model performance using accuracy and F1-score metrics.

---

## 🏗️ Model Architecture

### CRNN Architecture

Input:
- Log-Mel Spectrogram
- Shape: `(Batch Size, 1, 128, Time Frames)`

Feature Extraction:
- Conv2D → BatchNorm → ELU → MaxPooling
- Conv2D → BatchNorm → ELU → MaxPooling
- Conv2D → BatchNorm → ELU → MaxPooling

Temporal Modeling:
- 2-Layer Bidirectional GRU
- Hidden Size: 128

Classification Head:
- Fully Connected Layer
- ReLU Activation
- Dropout
- Output Layer

Output:
- Instrument Class Prediction

---

## 🔍 Dataset

The model is designed for musical instrument classification using audio datasets such as:

- IRMAS (Instrument Recognition in Musical Audio Signals)
- Custom instrument datasets

Audio files are transformed into:

- Log-Mel Spectrograms
- 128 Mel Frequency Bins

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-Learn
- Librosa
- Matplotlib
- Jupyter Notebook

---

## 📂 Project Structure

```text
music-instrument-recognition-crnn/
│
├── coursework.ipynb
├── instrument_crnn.py
├── trainer.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── checkpoints/
│   └── best_model.pt
│
└── results/
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/music-instrument-recognition-crnn.git
cd music-instrument-recognition-crnn
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```txt
torch
numpy
scikit-learn
librosa
matplotlib
pandas
jupyter
```

---

## 🚀 Training

Run the training script:

```bash
python trainer.py
```

Training features include:

- Cross Entropy Loss
- AdamW Optimizer
- Learning Rate Scheduling
- Early Stopping
- Model Checkpoint Saving
- Accuracy and F1-Score Tracking

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro F1-Score
- Classification Report
- Confusion Matrix

Example Metrics:

| Metric | Score |
|----------|---------|
| Accuracy | 90%+ |
| Macro F1 Score | 0.89+ |

*(Results may vary depending on dataset and training configuration.)*

---

## 🧠 Why CRNN?

CNNs excel at extracting local spectral patterns from spectrograms, while RNNs effectively model temporal relationships across audio frames.

Combining both architectures allows the model to:

✅ Learn frequency-based instrument characteristics

✅ Capture temporal evolution of musical notes

✅ Improve classification performance on real-world audio recordings

---

## 🔮 Future Improvements

- Transformer-based Audio Models
- Attention Mechanisms
- Audio Data Augmentation
- Multi-Task Learning
- Real-Time Instrument Recognition
- Deployment as a Web Application

---

## 📈 Applications

- Music Information Retrieval (MIR)
- Audio Content Analysis
- Smart Music Recommendation Systems
- Automatic Music Annotation
- Educational Music Software
- Digital Audio Workstations

---

## 👨‍💻 Author

**Sadaf Patel**

MSc Artificial Intelligence  
Queen Mary University of London

---

## ⭐ Acknowledgements

- PyTorch Team
- Librosa Developers
- Scikit-Learn Contributors
- IRMAS Dataset Creators
- Queen Mary University of London

If you found this project useful, consider giving the repository a ⭐ on GitHub.
