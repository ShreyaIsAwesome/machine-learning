# 🧬 Colorectal Cancer Tissue Classifier

This is a Flask-based web application that classifies tissue images from colorectal cancer slides into one of eight categories using a trained TensorFlow model.

[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)

## 🚀 Features

- Upload `.png` or `.jpg` images of colon tissue
- Get an instant classification and human-readable explanation
- Deep learning model trained on histopathological data
- Supports categories:
  - Tumor
  - Stroma
  - Complex
  - Lympho
  - Adipose
  - Mucosa
  - Debris
  - Empty

---

## 🧠 Model Description

The model is a convolutional neural network trained to recognize key tissue types in colorectal cancer slides. Understanding the presence and distribution of these tissues can help pathologists assess tumor behavior and patient prognosis.

Model is downloaded from Google Drive during app startup to avoid bloating the repo.

## 🖼 Tissue Labels Explained

Each prediction is accompanied by a description:

| Label     | Meaning |
|-----------|---------|
| `tumor`   | Cancer cells with abnormal structure |
| `stroma`  | Supporting tissue and blood supply |
| `complex` | Irregular, tangled gland structures |
| `lympho`  | Immune cells fighting cancer |
| `adipose` | Fat tissue around tumor |
| `mucosa`  | Healthy colon lining |
| `debris`  | Dead/dying cell fragments |
| `empty`   | Background/empty slide areas |

---

## ⚙️ Installation & Running

```bash
git clone https://github.com/your-username/colorectal-cancer-classifier.git
cd colorectal-cancer-classifier
pip install -r requirements.txt
python app.py
