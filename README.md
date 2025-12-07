# SSL-MEPR
SSL-MEPR: A Semi-Supervised Multi-Task Cross-Domain Learning Framework for Multimodal Emotion and Personality Recognition


## Text Modality Results
**Best-performing configurations**: BAAI/bge-small-en-v1.5 + Transformer and BAAI/bge-small-en-v1.5 + Mamba (for emotions) and Transformer (for personality)

### Evaluation Metrics (Test Set)
|Model              |Setup |CMU-MOSEI|CMU-MOSEI| FIv2    | FIv2    | Average  |
|-------------------|------|---------|---------|---------|---------|----------|
|                   |      | UAR     | MF1     | mACC    | mCCC    |          |
|Transformer        |  SD  | 64.69   | 59.47   | 88.90   | 26.83   |  59.97   |
|Transformer        |  MD  | 64.80   | 59.22   | 88.87   | 29.94   |  60.71   |
|Mamba + Transformer|  SD  | 65.02   | 59.78   | 88.90   | 26.83   |  60.13   |
|Mamba + Transformer|  MD  | 64.81   | 58.48   | 88.83   | 30.97   |  60.77   |

📦 [Download best model checkpoint](https://drive.google.com/drive/folders/1T8qcKY4SfC6135tquQ7V8TZeFQZy2WNZ?usp=sharing)
