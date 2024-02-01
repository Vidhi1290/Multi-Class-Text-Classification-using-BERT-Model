# Multi-Class Text Classification with BERT 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8%2B-orange)](https://pytorch.org/get-started/locally/)

## Overview

- Predict consumer financial product categories using BERT, a powerful pre-trained NLP model.
- Dataset: Over two million customer complaints.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, torch, nltk, sklearn, transformers

## Approach

1. **Data Processing**
   - Read CSV, handle null values, encode labels, preprocess text.

2. **Model Building**
   - Create BERT model, define dataset, train and test functions.

3. **Training**
   - Load data, split, create datasets and loaders.
   - Train BERT model on GPU/CPU.

4. **Predictions**
   - Make predictions on new text data.

## Project Structure

- **Input:** complaints.csv
- **Output:** bert_pre_trained.pth, label_encoder.pkl, labels.pkl, tokens.pkl
- **Source:** model.py, data.py, utils.py
- **Files:** Engine.py, bert.ipynb, processing.py, predict.py, README.md, requirements.txt

## Takeaways

1. Solving business problems using pre-trained models.
2. Leveraging BERT for text classification.
3. Data preparation and model training.
4. Making predictions on new data.

## Open for Project Collaboration

🤝 **Kindly connect on [LinkedIn](https://www.linkedin.com/in/vidhi-waghela-434663198/) and follow on [Kaggle](https://www.kaggle.com/vidhikishorwaghela). Let's collaborate and innovate together! 🌐✨
