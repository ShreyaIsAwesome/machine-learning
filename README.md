# Colorectal Cancer Tissue Classifier

This is a Flask-based web application that classifies tissue images from colorectal cancer slides into one of eight categories using a trained TensorFlow model. It uses a CNN (convolutionary nueral network) to identify patterns in the images inputed and then outputs the according label based on its classification. This is my first trained AI model and I hope to continue and make a lot more in the future.

## Tissue Labels Explained

Each prediction is accompanied by a description:

| Label     | Meaning |
|-----------|---------|
| `tumor`   | Cancer cells|
| `stroma`  | Supporting tissue >> for structure |
| `complex` | Irregular, tangled gland structures (looks very complicated)|
| `lympho`  | Immune cells fighting cancer (white blood cells)|
| `adipose` | Fat tissue around tumor |
| `mucosa`  | Healthy cell >> very organized (opposite of complex)|
| `debris`  | Dead/dying cell parts |
| `empty`   | Background/empty slide areas |

[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)
